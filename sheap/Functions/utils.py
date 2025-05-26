


# TODO add continium to gaussian sum as and option
def combine_auto(funcs):
    """
    Assumes each function 'f' has an attribute `f.n_params` that tells how many
    parameters it needs. Then automatically slices based on that.
    """

    def combined_func(x, all_args):
        start = 0
        total = 0
        for f in funcs:
            part_size = f.n_params  # e.g., if gauss.n_params = 3
            fargs = all_args[start : start + part_size]
            start += part_size
            total += f(x, fargs)
        return total

    return combined_func


def param_count(n):
    """
    A decorator that attaches an attribute `.n_params` to the function,
    indicating how many parameters it expects.
    """

    def decorator(func):
        func.n_params = n
        return func

    return decorator

def with_param_names(param_names: list[str]):
    def decorator(func):
        func.param_names = param_names
        func.n_params = len(param_names)
        return func
    return decorator