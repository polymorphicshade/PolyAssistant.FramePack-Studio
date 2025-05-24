def xy_plot_process_wrapper(settings, xy_plot_process, *args):
    """
    Wrapper function for xy_plot_process that gets the settings values from the settings object
    and passes them to the xy_plot_process function.
    
    Args:
        settings: The settings object
        xy_plot_process: The original xy_plot_process function
        *args: The arguments to pass to the xy_plot_process function
    
    Returns:
        The result of the xy_plot_process function
    """
    # Get the settings values
    gpu_memory_preservation_value = settings.get("gpu_memory_preservation", 6)
    mp4_crf_value = settings.get("mp4_crf", 16)
    
    # Insert the settings values at the appropriate positions (after xy_plot_rs)
    args_list = list(args)
    args_list.insert(20, gpu_memory_preservation_value)
    args_list.insert(21, mp4_crf_value)
    
    # Check if we're using the Generate tab's XY Plot
    # If args_list[22] (axis_x_switch) is "Nothing" and args_list[25] (axis_y_switch) is not "Nothing"
    # Then swap them to avoid the "For using Y axis, first use X axis" error
    if len(args_list) > 25:
        if args_list[22] == "Nothing" and args_list[25] != "Nothing":
            # Swap X and Y axis values
            args_list[22], args_list[25] = args_list[25], args_list[22]  # Swap switches
            args_list[23], args_list[26] = args_list[26], args_list[23]  # Swap text values
            args_list[24], args_list[27] = args_list[27], args_list[24]  # Swap dropdown values
    
    # Call the original function with the modified arguments
    return xy_plot_process(*args_list)
