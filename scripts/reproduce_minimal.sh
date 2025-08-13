#!/bin/bash
set -eoux pipefail

exp_dir=./exp/minimal
mkdir -p $exp_dir

# using all gpt-4-turbo for this minimal example
# model=gpt-4-1106-preview
# model_judge=gpt-4-1106-preview

# 使用本地 Llama-2-7b-chat-hf
# model=llama-2-7b-chat-hf
# model_judge=llama-2-7b-chat-hf

model=Llama-3.2-3b
# model_judge=Llama-3.2-3b
model_judge=llama-2-7b-chat-hf

# minimal example to reduce compute costs so using BoN=1 and only 10 questions
# if you want to run the full experiment, use BoN=16, temperature=0.8 and limit=47 (see run_figure1.sh)
BoN=1
temperature=0.4
limit=4
dataset_args="++max_num_from_same_story=5 ++split=both ++human_experiments=8 ++limit=$limit" # T_h dataset (47 questions)

# Run all experiments
debate_args="++correct_debater.language_model.model=$model \
            ++incorrect_debater.language_model.model=$model \
            ++correct_preference.language_model.model=$model \
            ++incorrect_preference.language_model.model=$model \
            ++correct_debater.BoN=$BoN ++incorrect_debater.BoN=$BoN \
            ++correct_debater.language_model.temperature=$temperature \
            ++incorrect_debater.language_model.temperature=$temperature \
            ++cross_examiner.language_model.model=$model_judge \
            ++anthropic_num_threads=1"
python -m core.debate exp_dir=$exp_dir +experiment='debate' $debate_args $dataset_args
python -m core.debate exp_dir=$exp_dir +experiment='debate_interactive' $debate_args $dataset_args
python -m core.debate exp_dir=$exp_dir +experiment='consultancy' method_type='correct' $debate_args $dataset_args
python -m core.debate exp_dir=$exp_dir +experiment='consultancy' method_type='incorrect' $debate_args $dataset_args
python -m core.debate exp_dir=$exp_dir +experiment='blind' $debate_args $dataset_args
python -m core.debate exp_dir=$exp_dir +experiment='oracle' $debate_args $dataset_args

# Judge all experiments
judge_args="++judge.language_model.model=$model_judge ++judge_name=$model_judge"
python -m core.judge exp_dir=$exp_dir +experiment='debate' $judge_args
python -m core.judge exp_dir=$exp_dir +experiment='debate_interactive' $judge_args
python -m core.judge exp_dir=$exp_dir +experiment='consultancy' method_type='correct' $judge_args
python -m core.judge exp_dir=$exp_dir +experiment='consultancy' method_type='incorrect' $judge_args
python -m core.judge exp_dir=$exp_dir +experiment='blind' $judge_args
python -m core.judge exp_dir=$exp_dir +experiment='oracle' $judge_args

# Score all experiments
score_args="++judge_name=$model_judge"
python -m core.scoring.accuracy exp_dir=$exp_dir +experiment='debate' $score_args
python -m core.scoring.accuracy exp_dir=$exp_dir +experiment='debate_interactive' $score_args
python -m core.scoring.accuracy exp_dir=$exp_dir +experiment='consultancy' method_type='correct' $score_args
python -m core.scoring.accuracy exp_dir=$exp_dir +experiment='consultancy' method_type='incorrect' $score_args
python -m core.scoring.accuracy exp_dir=$exp_dir +experiment='blind' $score_args
python -m core.scoring.accuracy exp_dir=$exp_dir +experiment='oracle' $score_args
