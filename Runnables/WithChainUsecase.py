import random 
class NakliLLM:
    def __init__(self):
        print("LLM Created")

    def predict(self,prompt):
        response_list=[
            "Sure, Delhi is capital of india",
            "Hii, How are you?",
            "Lets have a depp down discussion on this",
            "Have you had your dinner today?"
        ]

        return {"response": random.choice(response_list)}
    
model = NakliLLM()

class NakliProptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict) # This will replace the placeholders in the template with the values from input_dict
    
class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        formatted_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(formatted_prompt)
        return result['response']
    

template = NakliProptTemplate(template = "What is the capital of {country}?", input_variables = ['country'])
chain = NakliLLMChain(llm=model, prompt=template)
result = chain.run({"country": "India"})
print(result)

#drawback of using this approach is that we have to create a new class for every new chain we want to create and it
#  becomes difficult to manage and scale when we have multiple chains. Also, we have to manually handle the flow 
# of data between different chains which can lead to errors and bugs in the code.