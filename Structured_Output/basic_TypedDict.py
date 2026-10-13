from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

new_person:Person = {"name":"Madhu","age":"30"}#no error for age as typedDict doesn't validate types at runtime
print(new_person)