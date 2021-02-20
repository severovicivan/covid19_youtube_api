# Analysis of YouTube video comments related to coronavirus in 2020

Script **covid_extrakcija.py** extracts YouTube video comments for search query *'korona virus'* and generates ListaVidea.csv that is used in script **covid_analiza.py** for generating insights that are exposed and explained in **Severovic_Ivan_diplomski_rad.pdf**

2 graphs below shows that impact of anti-cov measures on users activity is bigger than impact of infected people number
![Measures vs Activity](https://github.com/severovicivan/covid19_youtube_api/blob/main/Screenshots/komentari_slucajevi.png)

But there still exists natural corellation between number of infected people and users activity for second corona wave
![Activity vs NOC](https://github.com/severovicivan/covid19_youtube_api/blob/main/Screenshots/pearsonova_korelacija.png)

Finally we see most commented videos for 2020 year and second wave (political parties that are against uneffective strict measures)
![Popular videos](https://github.com/severovicivan/covid19_youtube_api/blob/main/Screenshots/prvi_vs_drugi_val.png)
