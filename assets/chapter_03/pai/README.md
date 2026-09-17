# Chapter 3 PAI-Bench-G Video Review

These 56 videos are the original scored outputs. Their SHA-256 checksums match
the saved run indexes.

Compare the same case and seed across guidance settings. The case descriptions
below summarize the requested events in the
[benchmark prompts](../results/pai_subset_full_info.json). Read the questions
and judge answers alongside the videos, then record timestamps for the events
that support your interpretation.

| Case and requested event | Seed 0, guidance 7 | Seed 0, guidance 3 | Seed 1, guidance 7 | Seed 1, guidance 3 |
|---|---|---|---|---|
| Driving: approach a school crossing as cyclists and an oncoming sedan pass | [Open MP4](guidance7/videos/av_002b815b-7595-4d54-a9f4-4c62c83d3009__0.mp4) | [Open MP4](guidance3/videos/av_002b815b-7595-4d54-a9f4-4c62c83d3009__0.mp4) | [Open MP4](guidance7/videos/av_002b815b-7595-4d54-a9f4-4c62c83d3009__1.mp4) | [Open MP4](guidance3/videos/av_002b815b-7595-4d54-a9f4-4c62c83d3009__1.mp4) |
| Driving: slow behind a signaling van at a red light | [Open MP4](guidance7/videos/av_00d8e9d4-d669-42d9-8aa9-16a09a47b531__0.mp4) | [Open MP4](guidance3/videos/av_00d8e9d4-d669-42d9-8aa9-16a09a47b531__0.mp4) | [Open MP4](guidance7/videos/av_00d8e9d4-d669-42d9-8aa9-16a09a47b531__1.mp4) | [Open MP4](guidance3/videos/av_00d8e9d4-d669-42d9-8aa9-16a09a47b531__1.mp4) |
| Common sense: pack snacks into a paper cup with tongs | [Open MP4](guidance7/videos/common_sense_049c9292-db5f-490b-a72b-546775e8156c__0.mp4) | [Open MP4](guidance3/videos/common_sense_049c9292-db5f-490b-a72b-546775e8156c__0.mp4) | [Open MP4](guidance7/videos/common_sense_049c9292-db5f-490b-a72b-546775e8156c__1.mp4) | [Open MP4](guidance3/videos/common_sense_049c9292-db5f-490b-a72b-546775e8156c__1.mp4) |
| Common sense: take a bite of a burger, chew, and gesture | [Open MP4](guidance7/videos/common_sense_062f7b4e-1f09-4a1e-9f1a-5c531c9e2155__0.mp4) | [Open MP4](guidance3/videos/common_sense_062f7b4e-1f09-4a1e-9f1a-5c531c9e2155__0.mp4) | [Open MP4](guidance7/videos/common_sense_062f7b4e-1f09-4a1e-9f1a-5c531c9e2155__1.mp4) | [Open MP4](guidance3/videos/common_sense_062f7b4e-1f09-4a1e-9f1a-5c531c9e2155__1.mp4) |
| Human activity: pedestrians walk along a tree-lined street | [Open MP4](guidance7/videos/human_000__0.mp4) | [Open MP4](guidance3/videos/human_000__0.mp4) | [Open MP4](guidance7/videos/human_000__1.mp4) | [Open MP4](guidance3/videos/human_000__1.mp4) |
| Human activity: pedestrians cross the plaza around Cloud Gate | [Open MP4](guidance7/videos/human_001__0.mp4) | [Open MP4](guidance3/videos/human_001__0.mp4) | [Open MP4](guidance7/videos/human_001__1.mp4) | [Open MP4](guidance3/videos/human_001__1.mp4) |
| Industry: yellow crates travel around a curved conveyor | [Open MP4](guidance7/videos/industry_001__0.mp4) | [Open MP4](guidance3/videos/industry_001__0.mp4) | [Open MP4](guidance7/videos/industry_001__1.mp4) | [Open MP4](guidance3/videos/industry_001__1.mp4) |
| Industry: machinery moves and caps bottles on a conveyor | [Open MP4](guidance7/videos/industry_002__0.mp4) | [Open MP4](guidance3/videos/industry_002__0.mp4) | [Open MP4](guidance7/videos/industry_002__1.mp4) | [Open MP4](guidance3/videos/industry_002__1.mp4) |
| Miscellaneous: waves break against a rocky shore | [Open MP4](guidance7/videos/misc_1093662-hd_1920_1080_30fps__0.mp4) | [Open MP4](guidance3/videos/misc_1093662-hd_1920_1080_30fps__0.mp4) | [Open MP4](guidance7/videos/misc_1093662-hd_1920_1080_30fps__1.mp4) | [Open MP4](guidance3/videos/misc_1093662-hd_1920_1080_30fps__1.mp4) |
| Miscellaneous: geese and chicks forage beside a river | [Open MP4](guidance7/videos/misc_1200903-hd_1920_1080_30fps__0.mp4) | [Open MP4](guidance3/videos/misc_1200903-hd_1920_1080_30fps__0.mp4) | [Open MP4](guidance7/videos/misc_1200903-hd_1920_1080_30fps__1.mp4) | [Open MP4](guidance3/videos/misc_1200903-hd_1920_1080_30fps__1.mp4) |
| Physics: a released rubber duck falls into an open box | [Open MP4](guidance7/videos/physics_001__0.mp4) | [Open MP4](guidance3/videos/physics_001__0.mp4) | [Open MP4](guidance7/videos/physics_001__1.mp4) | [Open MP4](guidance3/videos/physics_001__1.mp4) |
| Physics: a tennis ball emerges from a pipe and rolls off-screen | [Open MP4](guidance7/videos/physics_002__0.mp4) | [Open MP4](guidance3/videos/physics_002__0.mp4) | [Open MP4](guidance7/videos/physics_002__1.mp4) | [Open MP4](guidance3/videos/physics_002__1.mp4) |
| Robotics: an arm grasps a handle and opens a cabinet | [Open MP4](guidance7/videos/robot_000__0.mp4) | [Open MP4](guidance3/videos/robot_000__0.mp4) | [Open MP4](guidance7/videos/robot_000__1.mp4) | [Open MP4](guidance3/videos/robot_000__1.mp4) |
| Robotics: an arm places toy sushi into a pot | [Open MP4](guidance7/videos/robot_001__0.mp4) | [Open MP4](guidance3/videos/robot_001__0.mp4) | [Open MP4](guidance7/videos/robot_001__1.mp4) | [Open MP4](guidance3/videos/robot_001__1.mp4) |

Each video filename contains its benchmark case ID and seed. Both configurations
use Cosmos-Predict2.5-2B revision
`0d37c7498f54cee3c599d438d895a0a4a8608064`, 93 frames, and 36 inference steps.
The generation records preserve the shared settings and case coverage.

| Saved evidence | Guidance 7 | Guidance 3 |
|---|---|---|
| Generation settings | [Record](../results/pai_guidance7_generation.json) | [Record](../results/pai_guidance3_generation.json) |
| Original run IDs and video hashes | [Run index](../results/pai_guidance7_run_index.jsonl) | [Run index](../results/pai_guidance3_run_index.jsonl) |
| Video quality scores | [Per-video scores and aggregates](../results/pai_guidance7_quality.json) | [Per-video scores and aggregates](../results/pai_guidance3_quality.json) |
| Requested-event scores | [Summary](../results/pai_guidance7_vqa_summary.json) | [Summary](../results/pai_guidance3_vqa_summary.json) |
| Questions, expected answers, and judge responses | [Detailed results](../results/pai_guidance7_vqa_detailed.json) | [Detailed results](../results/pai_guidance3_vqa_detailed.json) |

The [review index](review_index.json) connects the local media files to their
hashes and experiment records. Archived manifests live under
`guidance7/manifests/` and `guidance3/manifests/`; they preserve the original
run records, including paths from the generation machine. Use the review index
to locate files in this checkout.

Each guidance directory also includes an `evaluation/` directory with the
original evaluator revision, installed environment, lockfile, judge revision,
and vLLM compatibility patch. Keep those records with the score files when
reviewing or comparing this experiment.
