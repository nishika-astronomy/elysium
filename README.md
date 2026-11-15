# 🪐 ELYSIUM
### Mars Terrain Analysis & Exploration Planning System

**ELYSIUM** is a Python-based Mars exploration analysis system focused on Western Elysium Planitia, particularly the region surrounding NASA's InSight landing site. It integrates Martian orbital observations and elevation datasets to analyse surface terrain, investigate astronaut accessibility, compare potential exploration routes, and identify scientifically significant waypoints for future exploration.

---

## 🚀 Project Overview

Exploring Mars requires more than identifying scientifically interesting locations. Future exploration missions must also consider surface topography, elevation variations, terrain characteristics, and accessibility.

ELYSIUM addresses this challenge by integrating publicly available Martian orbital datasets with geospatial processing and visualisation techniques. The system aims to transform planetary observations into meaningful information that can support robotic and future human exploration planning.

## 🎯 Objectives

- Analyse the surface topography of Western Elysium Planitia.
- Discover and organise relevant Martian orbital observations.
- Integrate elevation and imaging datasets from multiple planetary science missions.
- Visualise Martian terrain through interactive 3D elevation models.
- Investigate terrain characteristics relevant to astronaut accessibility.
- Compare potential exploration routes.
- Identify scientifically significant locations for further investigation.

## 🛰️ Study Region

**Western Elysium Planitia, Mars**

The project focuses on the region surrounding the InSight landing site, an area of considerable scientific interest for understanding Martian geology, surface characteristics, and subsurface properties.

## 📡 Data Sources

ELYSIUM utilises publicly available planetary science datasets and mapping resources.

| Dataset / Mission | Application |
|---|---|
| Mars Orbital Data Explorer (ODE) | Discovery of orbital observations and metadata |
| MOLA | Martian elevation and topographic analysis |
| CTX | Regional surface imaging |
| HiRISE | High-resolution surface observations |
| THEMIS | Thermal and surface analysis |
| JPL Mars Trek | Supplementary planetary mapping and terrain exploration; partial data overlap with the study region |

**Data Coverage Note:** JPL Mars Trek was considered as a supplementary mapping resource. However, its available data only partially overlaps with the selected Western Elysium Planitia study region. Therefore, it is not treated as a complete regional data source.

## ⚙️ Technical Implementation

ELYSIUM is developed primarily using Python, with specialised libraries for scientific data processing, geospatial analysis, and interactive visualisation.

### Core Components

- **Data Discovery:** Identifies relevant orbital observations through the Mars Orbital Data Explorer.
- **Observation Catalogue:** Organises discovered datasets and associated metadata.
- **Footprint Processing:** Processes orbital observation footprints for spatial analysis.
- **CRS Validation:** Examines coordinate reference systems to ensure spatial compatibility.
- **Terrain Visualisation:** Uses MOLA elevation data to generate interactive 3D representations of Martian terrain.

### Technologies Used

- Python
- Streamlit
- NumPy
- Plotly
- NASA Planetary Science Data
- Geospatial processing tools

## 🗺️ Project Workflow

1. **Data Discovery:** Retrieve relevant orbital observations for the selected Martian region.
2. **Data Organisation:** Filter and catalogue observations based on their metadata.
3. **Spatial Processing:** Examine observation footprints and coordinate reference systems.
4. **Terrain Analysis:** Process MOLA elevation data to investigate regional topography.
5. **Visualisation:** Generate interactive 3D terrain representations.
6. **Exploration Assessment:** Investigate accessibility, potential exploration routes, and scientifically valuable waypoints using available terrain information.

## 🌌 Why ELYSIUM?

Named after Elysium Planitia, one of Mars's major volcanic plains, ELYSIUM represents an effort to make planetary science datasets more accessible and useful for exploration analysis.

By combining orbital observations, elevation data, computational processing, and geospatial visualisation, the project explores how existing Martian datasets can contribute to understanding the challenges of exploring another world.

## 🔭 Future Scope

Potential extensions include:

- Incorporating additional high-resolution orbital datasets.
- Improving terrain-based accessibility assessment.
- Integrating more detailed geological and thermal information.
- Enhancing route comparison and waypoint prioritisation.
- Expanding the analysis to larger Martian regions.
- Developing more comprehensive exploration-planning capabilities.

## 📚 References & Data Access

- [NASA Mars Orbital Data Explorer](https://ode.rsl.wustl.edu/mars/)
- [NASA Planetary Data System](https://pds.nasa.gov/)
- [NASA Mars Global Surveyor / MOLA](https://science.nasa.gov/mission/mars-global-surveyor/)
- [NASA HiRISE](https://hirise.lpl.arizona.edu/)
- [NASA Mars Reconnaissance Orbiter](https://mars.nasa.gov/mro/)
- [NASA JPL Mars Trek](https://trek.nasa.gov/mars/)

## 👩‍🚀 Project Information

| Detail | Information |
|---|---|
| Project Name | ELYSIUM |
| Challenge | NASA Space Apps Challenge 2026 |
| Development | Independent / Solo Project |
| Primary Language | Python |
| Domain | Planetary Science, Mars Exploration & Geospatial Analysis |

---

*Exploring Mars through data, one coordinate at a time.* 🚀
