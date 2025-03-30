from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os

"vanillaOVO/WizardCoder-Python-7B-V1.0"
#


# --------------------------
# Configuration and Settings
# --------------------------
class SimpleConfig:
    def __init__(self, model_name="starcoder2-3b", model_full_name="bigcode/starcoder2-3b"):
        self.learning_rate = 1e-8
        self.weight_decay = 0.1
        self.num_epochs = 1200
        self.max_new_tokens = 256
        self.max_length = 512 #3072
        self.max_context_len = 512 #3072
        self.temperature = 0.1,
        self.prompt_len = 512 # 738 # 829
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_full_name = model_full_name
        self.model = None
        self.optimizer = None
        self.init_token()
        current_src_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_src_file_dir, "saved_models",self.model_name,
                                       )  # Directory to save models and checkpoints
        self.cache_dir = os.path.join(current_src_file_dir, "cache", self.model_name)  # Cache directory for tokenized data
        self.context_len = 512  # Maximum sequence length
        self.num_batches = 1  # Batch size for DataLoader
        self.dataset_file = os.path.join(current_src_file_dir, 'manual_data_set',
                                       "ExactSampleTrainSample.pkl")
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def init_token(self, tokenizer=None):
        self.bos = '</s>' # tokenizer.bos_token
        self.eos = '</s>' # tokenizer.eos_token
        self.pad = '[PAD]' # tokenizer.pad_token
        self.mask = None # tokenizer.mask_token
        self.unknown = '</s>' # tokenizer.unk_token
        self.pad_side = 'right' # tokenizer.padding_side
        self.bsys, self.esys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.binst, self.einst = "[INST]", "[/INST]"
        self.breason, self.ereason = "[REASON]", "[/REASON]"
        self.banswer, self.eanswer = "[ANSWER]", "[/ANSWER]"

        # prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"
        # inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # <s> [INST] <<SYS>>
        # You are a helpful assistant.
        # <</SYS>>
        #
        # Howdy! [/INST]
        #
        # Well, howdy there! *adjusts cowboy hat* It's a pleasure to meet you! How can I help you today? Do you have any questions or tasks you'd like me to assist you with? Just let me

        return tokenizer

    def get_training_layers(self, model):
        # get first and last layer's parameters
        params = list(model.parameters())
        # merge first and last layer's parameters
        return params # params[0:2] + params[-2:]

    def save(self, epoch, model, optimizer):
        # Save checkpoint including optimizer state.
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(self.model_path, f"{self.model_name}_model_{epoch}.pth"))

    def _load(self, model_save_dir, force_new=False):
        files = os.listdir(model_save_dir)
        files = [f for f in files if f.endswith('.pth')]
        files = [f for f in files if '_' in f]
        # Load the original configuration
        config = AutoConfig.from_pretrained(self.model_full_name)
        config.max_context_len = self.max_context_len

        # Change the maximum context length
        config.max_position_embeddings = self.max_context_len
        self.model = AutoModelForCausalLM.from_pretrained(self.model_full_name,
                                                    torch_dtype=torch.float16,
                                                    config=config,
                                                     device_map="cuda")
        #self.model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_full_name)
        # self.init_token(tokenizer)
        prams_to_train = self.get_training_layers(self.model)
        self.optimizer = torch.optim.AdamW(
            prams_to_train, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if len(files) > 0 and not force_new:
            print(f"Loading model from {model_save_dir}")
            # order by epoch
            files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            # get the last file
            model_file = os.path.join(model_save_dir, files[-1])
            checkpoint = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return self.model, self.optimizer, tokenizer

    def load(self, force_new=False):
        # Load latest checkpoint.
        # get all file in the model path
        return self._load(self.model_path, force_new)


    def log_answer(self, error_text, answer, idx):
        print(error_text, answer, idx)