import os
import dataset
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import trainer
import chatbot

menu_options = {
    1: 'Modelo TechGenius',
    2: 'Modelo Transfer learning',
    0: 'Salir'
}

def print_menu():
    for key in menu_options.keys():
        print(key, ': ', menu_options[key])

print_menu()
option = int(input('Escoga un modelo con el que conversar: '))

while option != 0:
    if option == 1:
        # Modelo TechGenius
        print("Seleccionaste el Modelo TechGenius")
    elif option == 2:
        # Modelo Transfer learning
        model_path = os.path.dirname(os.path.realpath(__file__)) + "modelo.pt"
        if(!os.path.isfile(csv_path)):
            #train
            logging.getLogger("transformers").setLevel(logging.ERROR)

            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.add_special_tokens({"pad_token": "<pad>",
                                            "bos_token": "<BOS>",
                                            "eos_token": "<EOS>"})
            tokenizer.add_tokens(["<bot>:"])
            

            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model.resize_token_embeddings(len(tokenizer))

            model = model.to(device)
            
            
            data = dataset.load_dataset(os.path.dirname(os.path.realpath(__file__)), tokenizer)
            trainer.model_trainer(data, mode, model_path, device)
        
        logging.getLogger("transformers").setLevel(logging.ERROR)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"pad_token": "<pad>",
                                        "bos_token": "<BOS>",
                                        "eos_token": "<EOS>"})
        tokenizer.add_tokens(["<bot>:"])

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.resize_token_embeddings(len(tokenizer))

        model = model.to(device)
        
        estado = torch.load(model_path)
        model.load_state_dict(estado)
        model.eval()
        chatbot.converse(tokenizer, device, model)
        
    elif option == 0:
        # Salir
        print("Saliendo del programa...")
    else:
        os.system('cls')

