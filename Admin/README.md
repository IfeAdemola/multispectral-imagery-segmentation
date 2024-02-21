# MSc lifetime

Created: MG 2024-02-19

This readme shall keep tracks of meetings, outcomes and todos.
General admin docs related to the MSc thesis can be added to the folder.

## 2024-02-21
- the story about Amazon rainforest needs to change to some other cause Max has dropped out from the supervision
- TODO MG: speak to Greenspin to find another application and secure the data
- The MSc thesis proposal need updates
  - related work - build more on the 3 lit links in the announcement
  - from the current papers in the proposal:
    - SLIC Superpixel - important to understand and should remain
    - U-Net - standard in segmentation, needs to remain
    - SegNet - probably a useful baseline 
  - methodology - needs updating, should focus on 2 strategies
    - A) using existing pre-trained models
      - how to bring multi-spectral data (many channels) to 3 input-channel models
      - natural images have other properties than satellite data - how to tackle
    - B) pre-train own model
      - semi-supervised or unsupervised model training to get a reasonable feature extractor
      - methods for segmentation based on the learned latent representations, e.g. clustering (vector quantization?) in latent space
- TODO EA:
  - read the papers in the original topic announcements and prepare some summary for discussion
  - check the repo-link therein for recent work on unsupervised segmentation in the natural-image domain and try to understand the general approach
  - prepare some material for discussing the papers at our next meeting

**Timing:** The MSc thesis shall be finished within this semester. We shall not waste any more time and need to focus on getting it started in the next couple of weeks (1st week in March).