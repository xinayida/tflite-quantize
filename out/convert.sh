tflite_convert \
  --output_file=quantize_model.tflite \
  --graph_def_file=quantize_model.pb \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=input \
  --output_arrays=labels_softmax \
  --mean_values=128 \
  --std_dev_values=127