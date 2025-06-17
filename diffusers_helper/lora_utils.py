from pathlib import Path, PurePath
from typing import Dict, List, Optional, Union, Tuple
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
import torch

def load_lora(transformer: torch.nn.Module, lora_path: Path, weight_name: str) -> Tuple[torch.nn.Module, str]:
    """
    Load LoRA weights into the transformer model.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path: Path to the folder containing the LoRA weights file.
        weight_name: Filename of the weight to load.

    Returns:
        A tuple containing the modified transformer and the canonical adapter name.
    """
    
    state_dict = _fetch_state_dict(
        lora_path,
        weight_name,
        True,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None)

    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
    
    # should weight_name even be Optional[str] or just str?
    # For now, we assume it is never None
    # The module name in the state_dict must not include a . in the name
    # See https://github.com/pytorch/pytorch/pull/6639/files#diff-4be56271f7bfe650e3521c81fd363da58f109cd23ee80d243156d2d6ccda6263R133-R134
    adapter_name = str(PurePath(weight_name).with_suffix('')).replace('.', '_DOT_')
    if '_DOT_' in adapter_name:
        print(
            f"LoRA file '{weight_name}' contains a '.' in the name. " +
            'This may cause issues. Consider renaming the file.' +
            f" Using '{adapter_name}' as the adapter name to be safe."
        )
    
    # Check if adapter already exists and delete it if it does
    if hasattr(transformer, 'peft_config') and adapter_name in transformer.peft_config:
        print(f"Adapter '{adapter_name}' already exists. Removing it before loading again.")
        # Use delete_adapters (plural) instead of delete_adapter
        transformer.delete_adapters([adapter_name])
    
    # Load the adapter with the original name
    transformer.load_lora_adapter(state_dict, network_alphas=None, adapter_name=adapter_name)
    print(f"LoRA weights '{adapter_name}' loaded successfully.")
    
    return transformer, adapter_name

def unload_all_loras(transformer):
    """
    Completely unload all LoRA adapters from the transformer model.
    """
    if hasattr(transformer, 'peft_config') and transformer.peft_config:
        # Get all adapter names
        adapter_names = list(transformer.peft_config.keys())
        
        if adapter_names:
            print(f"Removing all LoRA adapters: {', '.join(adapter_names)}")
            # Delete all adapters
            transformer.delete_adapters(adapter_names)
            
            # Force cleanup of any remaining adapter references
            if hasattr(transformer, 'active_adapter'):
                transformer.active_adapter = None
                
            # Clear any cached states
            for module in transformer.modules():
                if hasattr(module, 'lora_A'):
                    if isinstance(module.lora_A, dict):
                        module.lora_A.clear()
                if hasattr(module, 'lora_B'):
                    if isinstance(module.lora_B, dict):
                        module.lora_B.clear()
                if hasattr(module, 'scaling'):
                    if isinstance(module.scaling, dict):
                        module.scaling.clear()
            
            print("All LoRA adapters have been completely removed.")
        else:
            print("No LoRA adapters found to remove.")
    else:
        print("Model doesn't have any LoRA adapters or peft_config.")
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return transformer

def set_adapters(
    transformer: torch.nn.Module,
    adapter_names: Union[List[str], str],
    weights: Optional[Union[float, List[float]]] = None,
):
    """
    Activates and sets the weights for one or more LoRA adapters.
    """
    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

    # Expand a single weight to apply to all adapters if needed
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)

    if len(adapter_names) != len(weights):
        raise ValueError(
            f"The number of adapter names ({len(adapter_names)}) does not match the number of weights ({len(weights)})."
        )

    # Replace any None weights with a default value of 1.0
    sanitized_weights = [w if w is not None else 1.0 for w in weights]

    # Dynamically get the class name to find the correct scaling function.
    transformer_class_name = transformer.__class__.__name__
    scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING.get(transformer_class_name)
    
    final_weights = []
    if scale_expansion_fn:
        print(f"Using scale expansion function for model class '{transformer_class_name}'...")
        for weight in sanitized_weights:
            expanded_weight = scale_expansion_fn(transformer, [weight])
            final_weights.append(expanded_weight[0])
    else:
        print(f"Warning: No scale expansion function found for '{transformer_class_name}'. Using raw weights.")
        final_weights = sanitized_weights

    set_weights_and_activate_adapters(transformer, adapter_names, final_weights)