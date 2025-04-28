# track reconstruction details

| Directory | Description |
|:----------|:------------|
|`utils/`| Contains utility scripts for track reconstruction and matching, including `match_tracks.py` (track matching logic) and `reconstruct_tracks.py` (generalized track reconstruction interface). These are used by clustering-based reconstruction algorithms. |
|`algorithm/`| Contains implementations of the primary clustering algorithms used for track reconstruction: `Wrangler` (DFS-based) and `DBSCAN`. |

---------------------------
