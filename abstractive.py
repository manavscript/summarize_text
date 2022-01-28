"""
Encoder-Decoder Architecture 

Research Paper: https://arxiv.org/pdf/1912.08777.pdf
Medium Post: https://towardsdatascience.com/how-to-perform-abstractive-summarization-with-pegasus-3dd74e48bafb
"""

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

class Abstractive():
    filename = None
    tokenizer = None
    model = None
    tokens = None
    summary = None

    def __init__(self, filename) -> None:
        self.filename = filename
    
    
    def create_models(self) -> None:
        self.tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    
    
    def create_tokens(self) -> None:
        with open(self.filename, mode="r", encoding="utf-8-sig") as f:
            text = f.read()
        print("Here")  
        self.tokens = self.tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
        print("Here")
        print(self.tokens)

    def summarize(self) -> None:
        self.summary = self.model.generate(**self.tokens)
    
    def decode(self) -> None:
        print(self.tokenizer.decode(self.summary[0]))  

if __name__ == "__main__":
    sample = Abstractive("text/text.txt")
    sample.create_models()
    sample.create_tokens()
    sample.summarize()
    sample.decode()
