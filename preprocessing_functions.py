def standard_preprocessing_function(X, normalization_function=None):
    # detection

    # crop

    # normalization
    if normalization_function is not None:
        return normalization_function(X)


AVAILABLE_PREPROCESSING_FUNCTIONS = {"standard_preprocessing_function": standard_preprocessing_function}