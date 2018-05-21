import json

def map_parameters(args):
    """
    Given a list of args between 0 and 1 expand them to their real value
    Given no args fetch default values and scale it down to 0-1 domain using
    min and max values.
    The parameter mapping file is a dict in the form:
    "param_name": [default,min,max]
    :param args:
    :type args:
    :return:
    :rtype: dict
    """
    with open("parameter_mapping.json") as f:
        params = json.load(f)
        if args:
            iter_args = iter(args)
            params = {k: next(iter_args) * (v[2] - v[1]) + v[1]
                      for k, v in params.items()}
        else:
            params = {k: (v[0] - v[1]) / (v[2] - v[1]) for k, v in
                      params.items()}
    return params

