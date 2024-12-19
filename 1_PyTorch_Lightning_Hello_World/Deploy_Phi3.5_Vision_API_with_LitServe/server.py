#from model_ao import Phi
from model import Phi
import litserve as ls

class Phi35Vision(ls.LitAPI):
    def setup(self, device):
        self.model = Phi(device)

    def decode_request(self, request):
        return self.model.apply_chat_template(request.messages)

    def predict(self, inputs, context):
        yield self.model(inputs, context)

    def encode_response(self, outputs):
        for output in outputs:
            yield {"role":"assistant", "content": self.model.decode_tokens(output)}


if __name__ == "__main__":
    api = Phi35Vision()
    server = ls.LitServer(api, spec = ls.OpenAISpec())
    server.run(port=8000)
