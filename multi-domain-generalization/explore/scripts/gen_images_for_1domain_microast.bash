
for domain in "photo" "sketch" "cartoon"; do
    for class in "dog" "elephant" "giraffe" "guitar" "horse" "house" "person"; do
        CUDA_VISIBLE_DEVICES=0 python gen_images_microast.py  \
        --content_dir ./DATA/pacs/images/${domain}/${class} \
        --style ./DATA/pacs/images/art_painting/ \
        --store_folder ./styled_images/microAST_1domain_multistyles/pacs/images/${domain}/${class} \
        --selected_domain art_painting
    done
done

# CUDA_VISIBLE_DEVICES=1 python all_seen_domain.py  --root ./styled_images/microAST  --dataset-config-file ../configs/datasets/dg/pacs_total.yaml  --resume false
