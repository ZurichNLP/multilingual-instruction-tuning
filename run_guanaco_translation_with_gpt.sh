#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# German
python translate_with_gpt.py \
    --input_file data/guanaco_train_mono_1k_en.json \
    --output_file data/guanaco_train_mono_1k_de.json \
    --tgt_lang "German" \
    --src_key "text" \
    --dataset_type "guanaco" \
    --model_name "gpt-3.5-turbo"

# French
python translate_with_gpt.py \
    --input_file data/guanaco_train_mono_1k_en.json \
    --output_file data/guanaco_train_mono_1k_fr.json \
    --tgt_lang "French" \
    --src_key "text" \
    --dataset_type "guanaco" \
    --model_name "gpt-3.5-turbo"

# Chinese
python translate_with_gpt.py \
    --input_file data/guanaco_train_mono_1k_en.json \
    --output_file data/guanaco_train_mono_1k_zh.json \
    --tgt_lang "Mandarin Chinese" \
    --src_key "text" \
    --dataset_type "guanaco" \
    --model_name "gpt-3.5-turbo"

# Spanish
python translate_with_gpt.py \
    --input_file data/guanaco_train_mono_1k_en.json \
    --output_file data/guanaco_train_mono_1k_es.json \
    --tgt_lang "Spanish" \
    --src_key "text" \
    --dataset_type "guanaco" \
    --model_name "gpt-3.5-turbo"

# Russian
python translate_with_gpt.py \
    --input_file data/guanaco_train_mono_1k_en.json \
    --output_file data/guanaco_train_mono_1k_ru.json \
    --tgt_lang "Russian" \
    --src_key "text" \
    --dataset_type "guanaco" \
    --model_name "gpt-3.5-turbo"

# Catalan
python translate_with_gpt.py \
    --input_file data/guanaco_train_mono_1k_en.json \
    --output_file data/guanaco_train_mono_1k_ca.json \
    --tgt_lang "Catalan" \
    --src_key "text" \
    --dataset_type "guanaco" \
    --model_name "gpt-3.5-turbo"

# # Swedish
# python translate_with_gpt.py \
#     --input_file data/guanaco_train_mono_1k_en.json \
#     --output_file data/guanaco_train_mono_1k_sv.json \
#     --tgt_lang "Swedish" \
#     --src_key "text" \
#     --dataset_type "guanaco" \
#     --model_name "gpt-3.5-turbo"

# # Norwegian Bokmål
# python translate_with_gpt.py \
#     --input_file data/guanaco_train_mono_1k_en.json \
#     --output_file data/guanaco_train_mono_1k_no.json \
#     --tgt_lang "Norwegian Bokmål" \
#     --src_key "text" \
#     --dataset_type "guanaco" \
#     --model_name "gpt-3.5-turbo"

# # Danish
# python translate_with_gpt.py \
#     --input_file data/guanaco_train_mono_1k_en.json \
#     --output_file data/guanaco_train_mono_1k_da.json \
#     --tgt_lang "Danish" \
#     --src_key "text" \
#     --dataset_type "guanaco" \
#     --model_name "gpt-3.5-turbo"
    
# # Bulgarian
# python translate_with_gpt.py \
#     --input_file data/guanaco_train_mono_1k_en.json \
#     --output_file data/guanaco_train_mono_1k_bg.json \
#     --tgt_lang "Bulgarian" \
#     --src_key "text" \
#     --dataset_type "guanaco" \
#     --model_name "gpt-3.5-turbo"

# # Icelandic
# python translate_with_gpt.py \
#     --input_file data/guanaco_train_mono_1k_en.json \
#     --output_file data/guanaco_train_mono_1k_is.json \
#     --tgt_lang "Icelandic" \
#     --src_key "text" \
#     --dataset_type "guanaco" \
#     --model_name "gpt-3.5-turbo"

# # Hindi
# python translate_with_gpt.py \
#     --input_file data/guanaco_train_mono_1k_en.json \
#     --output_file data/guanaco_train_mono_1k_hi.json \
#     --tgt_lang "Hindi" \
#     --src_key "text" \
#     --dataset_type "guanaco" \
#     --model_name "gpt-3.5-turbo"

# # Greek
# python translate_with_gpt.py \
#     --input_file data/guanaco_train_mono_1k_en.json \
#     --output_file data/guanaco_train_mono_1k_el.json \
#     --tgt_lang "standard modern Greek" \
#     --src_key "text" \
#     --dataset_type "guanaco" \
#     --model_name "gpt-3.5-turbo"

echo ""
echo "Done!"
echo ""