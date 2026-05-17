from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    # name: str
    # Passing default values 
    name: str = 'Nitish'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="Decimal value represent the CGPA of student")

new_student = {'name': 'Aryan', 'email':'abc@bf.com' , 'cgpa': 8.0}
new_student2 = {}

student = Student(**new_student) # type: ignore
print(student)
print(student.age)


# Convert this to Dict 
student_dict = dict(student)
print(student_dict['age'])

# Convert to JSON
student_json = student.model_dump_json()
print(student_json)