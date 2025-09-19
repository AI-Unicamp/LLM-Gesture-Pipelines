import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import torchvision
from typing import Union
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('./DiffuseStyleGesture/BEAT-TWH-main/process')
sys.path.append('./DiffuseStyleGesture/BEAT-TWH-main/process/WavLM')
from process_TWH_bvh import wavlm_init, load_audio

# Deshabilitar advertencias de transformaciones beta de torchvision
torchvision.disable_beta_transforms_warning()

def load_tsv_unclipped(tsvfile):
    sentence = []
    with open(tsvfile, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            if len(line) == 3:
                try:
                    start, end, raw_word = line
                    start = float(start)
                    end = float(end)
                    sentence.append([start, end, raw_word])
                except ValueError as e:
                    print(f"Warning: Line {i+1} in {tsvfile} has invalid values ({line}), skipping it: {str(e)}")
            else:
                print(f"Warning: Line {i+1} in {tsvfile} has {len(line)} columns instead of 3, skipping it.")
    return sentence

def get_word_embeddings_with_offset_mapping(
    sentence_text: str,
    tsv_words: list, # Lista de [(start, end, word), ...]
    tokenizer,
    model,
    device,
    dim_embedding: int,
    tsv_basename: str = "" # Opcional: para añadir contexto a los prints
) -> list:
    """
    Obtiene embeddings promediados para palabras de tsv_words usando offset mapping.
    Devuelve una lista de embeddings numpy del mismo tamaño que tsv_words.
    Rellena con ceros si hay errores para una palabra específica.
    Imprime SOLO advertencias/errores.
    """
    word_embeddings_list = [np.zeros(dim_embedding, dtype=np.float32) for _ in tsv_words]

    try:
        inputs = tokenizer(
            sentence_text,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping").cpu().numpy()[0]
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    except Exception as e:
        print(f"*** ERROR ({tsv_basename}): Exception during tokenization for sentence '{sentence_text[:50]}...': {e}")
        return word_embeddings_list

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0].cpu()
    except Exception as e:
        print(f"*** ERROR ({tsv_basename}): Exception getting hidden states for sentence '{sentence_text[:50]}...': {e}")
        return word_embeddings_list

    valid_indices = []
    special_tokens_mask = np.array(tokenizer.get_special_tokens_mask(input_ids[0].cpu().tolist(), already_has_special_tokens=True), dtype=bool)
    for i, offset in enumerate(offset_mapping):
        is_special = special_tokens_mask[i]
        is_empty_offset = (offset[0] == offset[1])
        if not is_special and not is_empty_offset:
            valid_indices.append(i)

    if not valid_indices:
        print(f"!!! WARNING ({tsv_basename}): No valid tokens found after filtering special tokens for sentence '{sentence_text[:50]}...'. Assigning all zeros.")
        return word_embeddings_list

    valid_offsets = offset_mapping[valid_indices]
    valid_embeddings = hidden_states[valid_indices]

    current_char_idx_in_sentence = 0
    token_idx_offset = 0

    for i, (start_time, end_time, raw_word) in enumerate(tsv_words):
        word_found_in_sentence = False
        word_start_char = -1
        word_end_char = -1

        try:
            search_start = current_char_idx_in_sentence
            word_start_char = sentence_text.index(raw_word, search_start)
            word_end_char = word_start_char + len(raw_word)
            word_found_in_sentence = True
        except ValueError:
            print(f"!!! WARNING ({tsv_basename}): Word '{raw_word}' (index {i}) NOT FOUND in sentence text starting from char index {current_char_idx_in_sentence}.")
            print(f"     Sentence: '{sentence_text}'")
            current_char_idx_in_sentence += len(raw_word) + 1
            continue

        relevant_token_indices_in_valid = []
        first_match_idx_in_valid = -1
        temp_token_idx_in_valid = token_idx_offset
        while temp_token_idx_in_valid < len(valid_offsets):
            token_start, token_end = valid_offsets[temp_token_idx_in_valid]
            overlaps = max(word_start_char, token_start) < min(word_end_char, token_end)
            if overlaps:
                relevant_token_indices_in_valid.append(temp_token_idx_in_valid)
                if first_match_idx_in_valid == -1:
                    first_match_idx_in_valid = temp_token_idx_in_valid
            elif token_start >= word_end_char and first_match_idx_in_valid != -1:
                break
            temp_token_idx_in_valid += 1

        if relevant_token_indices_in_valid:
            embeddings_to_average = valid_embeddings[relevant_token_indices_in_valid]
            word_embedding = torch.mean(embeddings_to_average, dim=0).numpy()
            word_embeddings_list[i] = word_embedding.astype(np.float32)
            if first_match_idx_in_valid != -1:
                token_idx_offset = first_match_idx_in_valid
        else:
            print(f"!!! WARNING ({tsv_basename}): No token offsets found overlapping with word '{raw_word}' (index {i}, chars [{word_start_char}:{word_end_char}]). Assigning zero embedding.")
            print(f"     Sentence: '{sentence_text}'")

        if word_found_in_sentence:
            current_char_idx_in_sentence = word_end_char
    
    return word_embeddings_list


def load_tsv(tsvpath: str, clip_length: int, tokenizer, model, device) -> Union[np.ndarray, None]:
    TARGET_FPS = 30.0
    tsv_basename = os.path.basename(tsvpath)
    print(f"\n--- Loading TSV: {tsv_basename} ---")
    sentence_data = load_tsv_unclipped(tsvpath)

    if not sentence_data:
        print(f"*** ERROR ({tsv_basename}): TSV file is empty or contains no valid lines. Skipping.")
        return None

    if clip_length <= 0:
        print(f"*** ERROR ({tsv_basename}): Invalid clip_length ({clip_length}) provided. Skipping.")
        return None

    oraciones = []
    current_oracion_words = []
    max_words_per_sentence = 25
    pause_threshold = 1.0
    line_warnings = 0

    for i, line_content in enumerate(sentence_data):
        if len(line_content) != 3:
            if line_warnings < 5:
                print(f"!!! WARNING ({tsv_basename}): Line {i+1} has {len(line_content)} columns, expected 3. Skipping line: {line_content}")
            line_warnings += 1
            continue
        start, end, word = line_content
        try:
            start_f = float(start)
            end_f = float(end)
        except ValueError:
            if line_warnings < 5:
                print(f"!!! WARNING ({tsv_basename}): Invalid start/end time in line {i+1}. Skipping word: {start}, {end}, {word}")
            line_warnings +=1
            continue

        current_oracion_words.append((start_f, end_f, word))
        is_last_word = (i == len(sentence_data) - 1)
        ends_with_punctuation = word.endswith(('.', '?', '!'))
        long_pause_follows = False
        if not is_last_word:
            next_valid_idx = i + 1
            while next_valid_idx < len(sentence_data):
                if len(sentence_data[next_valid_idx]) == 3:
                    try:
                        next_start = float(sentence_data[next_valid_idx][0])
                        if next_start - end_f > pause_threshold:
                            long_pause_follows = True
                        break
                    except ValueError: pass
                next_valid_idx += 1
        reached_max_words = len(current_oracion_words) >= max_words_per_sentence
        if is_last_word or ends_with_punctuation or long_pause_follows or reached_max_words:
            if current_oracion_words:
                oraciones.append(current_oracion_words)
                current_oracion_words = []
    if current_oracion_words:
        oraciones.append(current_oracion_words)
    if line_warnings >= 5:
        print(f"!!! WARNING ({tsv_basename}): Suppressed further line format/timing warnings for this file (total: {line_warnings}).")

    try:
        dim_embedding = model.config.hidden_size
    except AttributeError:
        print(f"*** ERROR ({tsv_basename}): Could not get hidden_size from model.config.")
        return None
    print(f"  Info ({tsv_basename}): Embedding dimension: {dim_embedding}")

    dim = dim_embedding + 2
    textfeatures = np.zeros([clip_length, dim], dtype=np.float32)
    textfeatures[:, -1] = 1.0

    print(f"  Info ({tsv_basename}): Processing {len(oraciones)} sentences...")
    total_words_processed = 0
    words_with_zero_embedding = 0

    for idx_oracion, oracion_words in enumerate(oraciones):
        if not oracion_words:
            continue

        sentence_text = ' '.join([word for _, _, word in oracion_words])
        word_embeddings = get_word_embeddings_with_offset_mapping(
            sentence_text, oracion_words, tokenizer, model, device, dim_embedding, tsv_basename=tsv_basename
        )

        for i, (start, end, raw_word) in enumerate(oracion_words):
            total_words_processed += 1
            embedding = word_embeddings[i]
            start_frame = int(start * TARGET_FPS)
            end_frame = int(end * TARGET_FPS)
            start_frame = max(0, start_frame)
            end_frame = min(clip_length, end_frame)

            if start_frame < end_frame:
                is_embedding_zero = np.all(embedding == 0)
                if is_embedding_zero:
                    print(f"  -> Assigning ZERO embedding to frames [{start_frame}:{end_frame}] for word '{raw_word}' in {tsv_basename} (check previous warnings for this sentence)")
                    words_with_zero_embedding += 1

                textfeatures[start_frame:end_frame, :dim_embedding] = embedding
                textfeatures[start_frame:end_frame, -2] = 1.0 if "#" in raw_word else 0.0
                textfeatures[start_frame:end_frame, -1] = 0.0
            else:
                print(f"!!! WARNING ({tsv_basename}): Word '{raw_word}' has zero/negative duration in frames [{start_frame}:{end_frame}]. No assignment.")

    summary_message = f"--- Finished loading TSV: {tsv_basename}. Processed {total_words_processed} words across {len(oraciones)} sentences."
    if words_with_zero_embedding > 0:
        summary_message += f" ({words_with_zero_embedding} words assigned zero embedding due to mapping issues)."
        print(f"!!! SUMMARY-WARN ({tsv_basename}): {words_with_zero_embedding} words ended up with zero embeddings. Please review previous warnings for this file.")
    print(summary_message)
    return textfeatures


def main(args):
    print(f"Available GPU: {torch.cuda.is_available()}")
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    splits = []
    if args.process_train:
        splits.append(('train', args.wav_path, args.txt_path, args.train_npy_path))
    if args.process_val:
        splits.append(('val', args.val_wav_path, args.val_txt_path, args.val_npy_path))
    if args.process_tst:
        splits.append(('tst', args.tst_wav_path, args.tst_txt_path, args.tst_npy_path))

    if not splits:
        print("Error: No splits selected for processing (train, val, tst).")
        return

    for _, _, _, output_path in splits:
        os.makedirs(output_path, exist_ok=True)
        print(f"Output directory ensured: {output_path}")

    # Cargar modelos una sola vez si es posible
    print("\nLoading models...")
    wavlm_model, cfg = wavlm_init(args.wavlm_path, device)
    print(f"** The WavLM model was successfully imported from: {args.wavlm_path} **")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, device_map="auto")
    print(f"** The LLM model was successfully imported from: {args.llm_model_path} **\n")

    for split, wav_path, txt_path, output_path in splits:
        print(f"\nProcessing {split} split...")
        
        wav_files = glob.glob(os.path.join(wav_path, '*.wav'))
        txt_files = glob.glob(os.path.join(txt_path, '*.tsv'))

        wav_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in wav_files}
        txt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in txt_files}

        common_names = sorted(list(set(wav_dict.keys()) & set(txt_dict.keys())))
        
        if args.limit_files is not None and args.limit_files > 0:
            common_names = common_names[:args.limit_files]
        
        print(f"Found {len(common_names)} file pairs to process in {split} split.")

        for name in common_names:
            print(f'--- Processing file: {name} ---')
            combined_filename = f"{name}_text_audio.npy"
            text_audio_output_file = os.path.join(output_path, combined_filename)
            
            if os.path.exists(text_audio_output_file):
                print(f"File already processed: {combined_filename}, skipping.")
                continue

            wav_file = wav_dict[name]
            audio_f = load_audio(wav_file, wavlm_model, cfg, device) # Simplificado

            txt_file = txt_dict[name]
            clip_len = audio_f.shape[0]
            tsv = load_tsv(txt_file, clip_len, tokenizer, model, device)
            if tsv is None:
                continue
            
            # Capturar dimensiones antes de concatenar para mayor claridad
            audio_dim = audio_f.shape[1]
            text_dim = tsv.shape[1]

            textaudio = np.concatenate((audio_f, tsv), axis=-1)
            print(f"Dimension of embeddings (clip length, audio + texto): {textaudio.shape}, file: {combined_filename}")
            print(f" Detail: Audio={audio_dim}, Text={text_dim}, Total={textaudio.shape[1]}")
            np.save(text_audio_output_file, textaudio)
            print(f"Combined embeddings saved in: {text_audio_output_file}\n")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Procesar embeddings de audio y texto.')
    
    # CORRECCIÓN 2: Rutas por defecto ajustadas para ejecución desde /workspace
    # Model Paths
    parser.add_argument('--wavlm_path', type=str, default='models/WavLM/WavLM-Large.pt', help='Ruta al modelo WavLM.')
    parser.add_argument('--llm_model_path', type=str, default='models/llama-3.2-3b-instruct', help='Ruta al modelo LLM.')
    
    # Train Paths
    parser.add_argument('--wav_path', type=str, default='data/trn/main-agent/wav/', help='Ruta a los WAV de entrenamiento.')
    parser.add_argument('--txt_path', type=str, default='data/trn/main-agent/tsv/', help='Ruta a los TSV de entrenamiento.')
    parser.add_argument('--train_npy_path', type=str, default='data/trn/main-agent/text-audio/', help='Directorio de salida para .npy de entrenamiento.')
    
    # Validation Paths
    parser.add_argument('--val_wav_path', type=str, default='data/val/main-agent/wav/', help='Ruta a los WAV de validación.')
    parser.add_argument('--val_txt_path', type=str, default='data/val/main-agent/tsv/', help='Ruta a los TSV de validación.')
    parser.add_argument('--val_npy_path', type=str, default='data/val/main-agent/text-audio/', help='Directorio de salida para .npy de validación.')

    # Test Paths

    parser.add_argument('--tst_wav_path', type=str, default='data/tst/main-agent/wav/', help='Ruta a los WAV de test.')
    parser.add_argument('--tst_txt_path', type=str, default='data/tst/main-agent/tsv/', help='Ruta a los TSV de test.')
    parser.add_argument('--tst_npy_path', type=str, default='data/tst/main-agent/text-audio/', help='Directorio de salida para .npy de test.')
    
    # Processing Flags
    parser.add_argument('--process_train', action='store_true', help='Procesar el split de entrenamiento.')
    parser.add_argument('--process_val', action='store_true', help='Procesar el split de validación.')
    parser.add_argument('--process_tst', action='store_true', help='Procesar el split de test.')
    parser.add_argument('--limit_files', type=int, default=None, help='Número máximo de archivos a procesar por split.')
    
    args = parser.parse_args()
    main(args)


    
