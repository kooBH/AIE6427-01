import sys, time, os, argparse
sys.path.append("./model")
import numpy as np

parser = argparse.ArgumentParser(description = "SpeakerNet")

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

args = parser.parse_args()


if __name__ == "__main__" :
    import torch
    import torch.onnx
    import torch.nn as nn
    import RawNet3 as RawNet3

    # Load model
    model  = (RawNet3.MainModel(
        #nOut=512,
        nOut=256,
        encoder_type="ECA",
        sinc_stride=10,
        max_frame = 200,
        sr=16000
        ))
    model.load_state_dict(torch.load("./chkpt/model.pt")["model"])
    #model.load_state_dict(torch.load("./chkpt/model.pt"))
    model.eval()
    #torch.save(model.state_dict(),"./chkpt/rawnet3.pt",)

    # torch to ONNX

    ## tracing
    input = torch.rand(2,32000)
    output = model(input)
    print("{} -> {}".format(input.shape,output.shape))
    """
    traced_model = torch.jit.trace(model, input)
    traced_model.save('./chkpt/rawnet3_traced.pt')
    torch.backends.cudnn.deterministic = True
    """
    torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL)

    print("ONXX Export")
    torch.onnx.export(
        model,         # model being run 
        input,       # model input (or a tuple for multiple inputs) 
        "./chkpt/rawnet3.onnx",       # where to save the model  
        opset_version=12,
        do_constant_folding=False,
        keep_initializers_as_inputs=False,
        input_names = ['input'], 
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size',1: 'n_sample'},    # variable length axes
        'output' : {0 : 'batch_size'}},
        export_params=True,
        #verbose = True,
        training = False

        )

    print("== Test ==")
    import onnx
    import onnxruntime
    print(onnxruntime.get_device())

    print("ONNX::load model")
    onnx_model = onnx.load("./chkpt/rawnet3.onnx")

    print("ONNX::check model")
    onnx.checker.check_model(onnx_model)

    print("ONNX::inference session")
    ort_session = onnxruntime.InferenceSession("./chkpt/rawnet3.onnx",
    providers=["CUDAExecutionProvider"]
    )


    x = np.load("input.npy").astype(np.float32)

    print("ONNX run")
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    np.save("onnx.npy",ort_outs[0])
    print(ort_outs[0].shape)

    print("torch run")

    model = model
    torch_out = model(torch.from_numpy(x).float())
    torch_out = torch_out.detach().numpy()
    np.save("pytorch.npy",torch_out)
    print(torch_out.shape)
    
    np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-03, atol=1e-03)
    
    print("PASS!!")