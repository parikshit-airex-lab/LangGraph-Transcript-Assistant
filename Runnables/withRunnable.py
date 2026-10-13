from abc import ABC, abstractmethod
class runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass

import random
class NakliLLM(runnable):
    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt):
        response_list=[
            "Sure, Delhi is capital of india",
            "Hii, How are you?",
            "Lets have a depp down discussion on this",
            "Have you had your dinner today?"
        ]

        return {"response": random.choice(response_list)}
    
    def predict(self,prompt):
        response_list=[
            "Sure, Delhi is capital of india",
            "Hii, How are you?",
            "Lets have a depp down discussion on this",
            "Have you had your dinner today?"
        ]

        return {"response": random.choice(response_list),"Note": "This method is depreciated"}
    
class NakliPromptTemplate(runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict) # This will replace the placeholders in the template with the values from input_dict
    
    def format(self, input_dict):
        print("This method is depreciated, please use invoke method instead")
        return self.template.format(**input_dict) # This will replace the placeholders in the template with the values from input_dict
    

class NakliStrOutputParser(runnable):
    def __init__(self):
        print("parser created")

    def invoke(self, input_data):
        return input_data['response']
    
class RunnableConnectors(runnable):
    def __init__(self, runnable_list):
       self.runnable_list = runnable_list

    def invoke(self,input_data):
       for runnable in self.runnable_list:
           input_data = runnable.invoke(input_data)
       return input_data
    
llm = NakliLLM()
template = NakliPromptTemplate(
    template="Write a short notes on {topic}",
    input_variables=['topic']
)
template1 = NakliPromptTemplate(
    template="Write a quiz on {response}",
    input_variables=['response']
)


parser = NakliStrOutputParser()
chain1 = RunnableConnectors([template,llm])
#result = chain1.invoke({'topic':"India"})
chain2 = RunnableConnectors([template1,llm,parser])
#result2 = chain2.invoke({'notes': result})
final_chain = RunnableConnectors([chain1,chain2])
result3 = final_chain.invoke({'topic': "India"})
print(result3)