import onnx
import numpy as np
import onnxruntime

torch_out = np.load("pytorch.npy").astype(np.float32)
trt_out = np.load("TRT.npy").astype(np.float32)
onnx_out = np.load("onnx.npy").astype(np.float32)

error_level_rtol = 1e-03
error_level_atol = 1e-03

print("Relative Tolerance : {}".format(error_level_rtol))
print("Absolute Tolerance : {}".format(error_level_atol))

print("TESTING torch vs ONNX")
try :
    np.testing.assert_allclose(torch_out, onnx_out, rtol=error_level_rtol, atol=error_level_atol,verbose=True)
except AssertionError :
    print("ERROR")

print("TESTING torch vs TRT ")
try :
    np.testing.assert_allclose(torch_out, trt_out, rtol=error_level_rtol, atol=error_level_atol,verbose=True)
except AssertionError :
    print("ERROR")

print("TESTING TRT vs ONNX ")
try :
    np.testing.assert_allclose(trt_out, onnx_out, rtol=error_level_rtol, atol=error_level_atol,verbose=True)
except AssertionError :
    print("ERROR")