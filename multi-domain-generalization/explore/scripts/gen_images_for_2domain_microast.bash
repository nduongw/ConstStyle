
# for domain in "sketch"; do
#     for class in "dog" "elephant" "giraffe" "guitar" "horse" "house" "person"; do
#         CUDA_VISIBLE_DEVICES=0 python gen_images_microast.py  \
#         --content_dir ./DATA/pacs/images/${domain}/${class} \
#         --style ./DATA/pacs/images/cartoon/ \
#         --store_folder ./styled_images/microAST_2domain/pacs/images/${domain}/${class} \
#         --target-domains cartoon \
#         --source-domains art_painting sketch photo
#     done
# done

# for domain in "photo"; do
#     for class in "dog" "elephant" "giraffe" "guitar" "horse" "house" "person"; do
#         CUDA_VISIBLE_DEVICES=0 python gen_images_microast.py  \
#         --content_dir ./DATA/pacs/images/${domain}/${class} \
#         --style ./DATA/pacs/images/art_painting/ \
#         --store_folder ./styled_images/microAST_2domain/pacs/images/${domain}/${class} \
#         --target-domains art_painting \
#         --source-domains cartoon sketch photo
#     done
# done

CUDA_VISIBLE_DEVICES=0 python all_seen_domain.py  --root ./styled_images/microAST_2domain  --dataset-config-file ../configs/datasets/dg/pacs_total.yaml  --resume false