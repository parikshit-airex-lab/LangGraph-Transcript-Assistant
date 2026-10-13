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
    
template = NakliProptTemplate(template = "What is the capital of {country}?", input_variables = ['country'])
formatted_prompt = template.format({'country': 'India'})
result = model.predict(formatted_prompt)
print(result)