# 🪐 Download ELYSIUM

Follow these instructions to download the complete ELYSIUM project, including its scientific datasets.

## Prerequisites

Make sure you have installed:

- [Git](https://git-scm.com/downloads)
- [Git LFS](https://git-lfs.com/)

## 1. Clone the Repository

Open Git Bash or your terminal and execute:

```bash
git clone https://github.com/nishika-astronomy/elysium.git
```

Navigate into the downloaded project:

```bash
cd elysium
```

## 2. Initialise Git LFS

Configure Git Large File Storage:

```bash
git lfs install
```

## 3. Download the Scientific Datasets

ELYSIUM uses Git LFS to store its large scientific datasets.

Download the actual dataset files using:

```bash
git lfs pull
```

This retrieves the datasets required by ELYSIUM, including available MOLA, CTX, HiRISE, THEMIS, and processed observation data.

**Note:** The repository contains approximately 2.3 GB of data. Ensure you have sufficient disk space and a stable internet connection.

## 4. Verify the Download

Check the downloaded files:

```bash
git lfs ls-files
```

Your ELYSIUM project and datasets should now be available locally.

---

## Repository

**GitHub:** https://github.com/nishika-astronomy/elysium

For issues or questions, please visit the repository's Issues section.

*Explore Mars. Discover the possibilities.* 🚀
