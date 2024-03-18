import json
from typing import Any, Type, TypeVar, get_type_hints, Dict,  get_args, get_origin
import inspect

T = TypeVar('T')


def conforms_to_type(obj: Any, type_desc: Type[Any]) -> bool:
    """Check if 'obj' conforms to 'type_desc'."""
    # Direct type check for non-generic types
    if get_origin(type_desc) is None:
        return isinstance(obj, type_desc)

    origin = get_origin(type_desc)
    args = get_args(type_desc)

    if origin is list:
        # Check if obj is a list and all elements conform to the specified type
        return isinstance(obj, list) and all(conforms_to_type(item, args[0]) for item in obj)
    elif origin is dict:
        # Check if obj is a dict and all keys and values conform to the specified types
        expected_key_type, expected_value_type = args
        return isinstance(obj, dict) and all(
            conforms_to_type(key, expected_key_type) and
            conforms_to_type(value, expected_value_type) for key, value in obj.items())
    else:
        raise TypeError(f"WTF is this compound type expected by someone's from_dict() method? {type_desc}")

def class_instance_from_json(cls: Type[T], json_str: str) -> T:
    """
    Deserialize JSON into an object of the specified class
    with added type conformance checking based on the from_dict signature.

    :param cls: The class to deserialize the JSON into, which must have a `from_dict` class method.
    :param file_path: Path to the JSON file to deserialize.
    :return: An instance of `cls`.
    """
    if not hasattr(cls, 'from_dict') or not callable(getattr(cls, 'from_dict')):
        raise AttributeError("Class must have a callable from_dict class method")

    from_dict_method = getattr(cls, 'from_dict')
    # Get expected type hint for the first parameter of from_dict (excluding 'cls')
    param_type_hint = next(iter(get_type_hints(from_dict_method).values()), None)
    if param_type_hint is None:
        param_type_hint = Dict

    # get the expected return type hint
    return_type_hint = get_type_hints(from_dict_method).get('return', None)
    if return_type_hint and not return_type_hint ==  cls:
        raise TypeError(f"The return type hint of class method {cls.__name__}.from_dict(cls,d:dict) must be {cls} but it is {return_type_hint}")

    # Load the JSON data
    the_dict:dict = json.loads(json_str)

    # Validate the loaded data against the expected type hint
    if not conforms_to_type(the_dict, param_type_hint):
        raise TypeError(f"Loaded JSON data does not conform to the expected type: {param_type_hint}")

   # cannot do the following because the type is dynamically determined.
   # typed_data = cast(param_type_hint, data)

    # Deserialize the data into an instance of the class
    obj:T =  cls.from_dict(the_dict)  # type: ignore (we've already done runtime checks that cls has a from_dict method )
    return obj



def class_instance_to_json(instance: object) -> str:
    if not (hasattr(instance, 'to_dict') and callable(getattr(instance, 'to_dict'))):
        raise ValueError("instance must have the method to_dict(self).")

    to_dict_method = instance.to_dict # type: ignore (just checked that it does indeed have that method)

        # Inspect the signature of the to_dict method
    to_dict_sig = inspect.signature(to_dict_method)
    # Filter out 'self' from parameters
    params = [param for param in to_dict_sig.parameters.values() if param.name != 'self']

    if params:
        # If there are any parameters other than 'self', raise an error
        raise ValueError("to_dict method must take no arguments other than 'self'")

    the_dict = to_dict_method()
    if not (isinstance(the_dict,dict)):
        raise ValueError(f"instance.to_dict()  must return a dict but it returned this: {the_dict}")
    return json.dumps(the_dict)

