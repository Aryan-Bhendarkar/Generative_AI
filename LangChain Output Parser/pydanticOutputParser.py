from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from  pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct") # type: ignore
model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description = "Name of the Person")
    age: int = Field(gt=18, description="Age of the Person")
    city: str = Field(description="Name of the city person belong to ")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and the city of fictional {place} person \n {format_instruction}", 
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place': 'indian'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content) # type: ignore
chain = template | model | parser
final_result = chain.invoke({'place': 'Indian'})
print(final_result)