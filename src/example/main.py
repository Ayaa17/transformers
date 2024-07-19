from transformers import WhisperProcessor, TFWhisperForConditionalGeneration
import tensorflow as tf
from datasets import load_dataset, Audio


def convert_model(model_name="openai/whisper-tiny"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = TFWhisperForConditionalGeneration.from_pretrained(model_name)

    ds = load_dataset("mozilla-foundation/common_voice_11_0", "zh-TW", split="validation", use_auth_token=False,
                      trust_remote_code=True)
    common_voice = ds.cast_column("audio", Audio(sampling_rate=16000))

    print(f"ds: {common_voice[62]}")
    inputs = processor(common_voice[62]["audio"]["array"], return_tensors="tf")
    input_features = inputs.input_features

    class ConvertModel(tf.Module):
        def __init__(self, model):
            super(ConvertModel, self).__init__()
            self.model = model

        @tf.function(
            input_signature=[
                tf.TensorSpec((1, 80, 3000), tf.float32, name="input_ids"),
            ],
        )
        def serving(self, input_ids):
            outputs = self.model.generate(
                input_ids,
                return_dict_in_generate=True,
                forced_decoder_ids=[[1, 50260], [2, 50359], [3, 50363]],
            )
            return {"sequences": outputs["sequences"]}

    generate_model = ConvertModel(model=model)
    predicted_ids = generate_model.serving(input_features)["sequences"]
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print("Tensorflow Transcription:", transcription)

    converter = tf.lite.TFLiteConverter.from_keras_model(generate_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_path = 'whisper.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(tflite_model_path)
    tflite_generate = interpreter.get_signature_runner()
    generated_ids = tflite_generate(input_ids=input_features)["sequences"]
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(f"generated_ids: {generated_ids}")
    print(f"tflite transcription: {transcription}")


def test_tflite(model_name="openai/whisper-tiny", tflite_model_path='whisper.tflite'):
    processor = WhisperProcessor.from_pretrained(model_name)

    ds = load_dataset("mozilla-foundation/common_voice_11_0", "zh-TW", split="validation", use_auth_token=False,
                      trust_remote_code=True)
    common_voice = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"ds: {common_voice[62]}")
    inputs = processor(common_voice[62]["audio"]["array"], return_tensors="tf")
    input_features = inputs.input_features

    interpreter = tf.lite.Interpreter(tflite_model_path)
    tflite_generate = interpreter.get_signature_runner()
    generated_ids = tflite_generate(input_ids=input_features)["sequences"]
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(f"{tflite_model_path} - generated_ids: {generated_ids}")
    print(f"{tflite_model_path} - transcription: {transcription}")


if __name__ == '__main__':
    convert_model()
    # test_tflite(tflite_model_path='un_fix_whisper.tflite')
    # test_tflite(tflite_model_path='fixed_whisper.tflite')
