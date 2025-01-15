# Projecte d'electroencefalogrames
Aquest projecte se centra en, la majoria de casos, reutilitzar i adaptar codis existents per dur a terme les següents tasques:
- Classificació individual de totes les fonts per pacient
- Extracció de fonts comunes per grups de participants
- Classificació individual amb fonts comunes
- Comparació entre grups amb fonts comunes

## Estructura del projecte
Degut a la complexitat del projecte, s'han creat diverses carpetes i subcarpetes. A continuació, s'explica l'estructura principal i el contingut més rellevant:
- **pacients_PD**: Inclou dades i codis específics dels participants amb Parkinson.
- **pacients_sans**: Conté dades i codis específics dels participants sans.
- **matlab_programs_PD**: Inclou programes MATLAB específics per als participants amb Parkinson.
- **matlab_programs_sans**: Inclou programes MATLAB específics per als participants sans.
- **matlab_programs**: Inclou programes MATLAB no específics d’un grup de participants concret.


## Requisits
- **Python**: Versió 3.12.
- **MATLAB**: Versió 2023b, amb el paquet SPM12 per utilitzar la toolbox FieldTrip.

## Explicació del codi

### Classificació individual de totes les fonts per pacient
En aquest primer pas, es classifiquen les fonts de cada participant. Els programes utilitzats són:
- `classif_sans_SF.py`: Classifica participants sans segons la motivació SF.
- `classif_sans_SP.py`: Classifica participants sans segons la motivació SP.
- `classif_PD_SF.py`: Classifica participants amb Parkinson segons la motivació SF.
- `classif_PD_SP.py`: Classifica participants amb Parkinson segons la motivació SP.

Per executar qualsevol d'aquests quatre programes s'ha d'introduir a terminal
```bash
python <nom_programa> i_sub i_motion
```
on i_sub és el número del pacient i i_motion el tipus de moviment (0 per NS i 1 per S)

Els resultats es guarden a la carpeta `results` dins de les carpetes respectives (`pacients_sans` o `pacients_PD`), amb subcarpetes estructurades com:
- **Participants sans**: `sub<num_sub>-<motiv>-<motion>-ICA`, on `<num_sub>` representa el número del participant,
`<motiv>` el tipus de motivació (SF o SP), `<motion>` el tipus de moviment realitzat (NS per Cross-Over, S per Stop-In i ALL per no
diferenciar el tipus de moviment).
- **Participants amb Parkinson**: `sub<num_sub>-<motiv>-<motion>-ICA-<med>` (`<med>` pot ser ON o OFF).

Els resultats inclouen:
- Imatges amb les precisions per banda de freqüència i mètrica (potència o correlació).
- Matrius de confusió.

Per combinar els resultats es fa servir el programa `plot_accuracies.py`, que genera violin plots i matrius de confusió. Els resultats es guarden a les carpetes `results_SF` i `results_SP`.

Per executar el programa només és necessari introduir a terminal
```bash
python plot_accuracies.py
```

El nom de les imatges que s'han guardat a través d'aquest programa és el següent:
- accuracy_<tipus_classif>_<classificador>.pdf o accuracy_<tipus_classif>_<classificador>.png
- conf_matrix_<tipus_classif>_<classificador>.pdf o conf_matrix_<tipus_classif>_<classificador>.png
on <tipus_classif> és pow o corr segons si les dades a classificar eren segons la potència de les fonts o la correlació
i <classificador> és MLR o 1NN.

### Extracció de fonts comunes per grups de participants
Per poder extreure les fonts comunes de cada grup de participants, primer de tot s'han utilitzat uns scripts de MATLAB. Aquests scripts també s'han hagut d'adaptar segons si els participants eren de Parkinson o no. Per tant:

- **Participants amb Parkinson**: 
  - S'ha utilitzat el programa `plotComponents_v1.m` ubicat dins de la carpeta `matlab_programs_PD`.
  - Aquest programa crida al programa `getSourceCoefficientOnElectrodes.m` i utilitza el fitxer `GSN129.sfp` per trobar la localització de cada font per a cada participant.
  - Els resultats d'aquesta localització es guarden a la subcarpeta `sources` dins de la carpeta `pacients_PD`.

- **Participants sans**: 
  - S'ha utilitzat el mateix programa, `plotComponents_v1.m`, però ubicat dins de la carpeta `matlab_programs_sans`.
  - En aquest cas, s'ha utilitzat el fitxer `ActiCap64_LM.lay` en lloc de `GSN129.sfp`. Aquest fitxer especifica la localització de cada elèctrode durant l'experiment.
  - Els resultats s'han guardat a la subcarpeta `sources` dins de la carpeta `pacients_sans`.

Per executar aquesta part es fa des de MATLAB. Només cal executar el fitxer `plotComponents_v1.m` (corresponent a cada carpeta).

Un cop trobada la localització de cada font (source), s'han comparat les fonts de cada grup. Per tant
s'ha usat un programa anomenat `calc_common_ICAs.py`, que està a les carpetes pacients_sans i pacients_PD segons
el tipus de participant a analitzar. Els resultats s'han guardat a:
- **Sans**: `common_src_SF` i `common_src_SP`.
- **Parkinson**: `common_src_<motiv>_<med>`.

Per executar el programa només és necessari introduir a terminal (des de dins de la carpeta corresponent)
```bash
python calc_common_ICAs.py
```

En el cas del dels participants amb Parkinson s'ha d'esmentar que per trobar les fonts comunes s'han separat les dades segons 
si el pacient estava medicat o no i s'han fet comparacions separades.

### Classificació individual amb fonts comunes
En aquest cas, s'ha realitzat una tasca similar a l'apartat de classificació individual amb totes les fonts, però ha estat necessari implementar un programa diferent. Els programes utilitzats són:

- `classif_freq_common_ICAs_<motiv>.py`: on `<motiv>` pot ser SF o SP. Aquests programes es troben dins de les carpetes `pacients_sans` i `pacients_PD` i per executar-los només cal introduir `python <nom_programa>`.

Per tal de veure els resultats generals comparant participants, s'ha implementat el programa `plot_accuracies_common_src.py`. 

Els resultats s'han guardat, novament, a les carpetes `results_SF` i `results_SP`.


### Comparació entre grups amb fonts comunes
Per realitzar aquesta última comparació, s'han utilitzat els resultats obtinguts de l'extracció de fonts comunes de cada grup. 
Per comparar aquestes fonts comunes entre grups, s'ha implementat el programa `calc_common_ICAs_all_mean.py`. 
Aquesta comparació s'ha fet en parelles de grups: ON vs OFF, sans vs ON i sans vs OFF.

Finalment, per visualitzar les fonts comunes de cada grup i les fonts resultants de la comparació, s'ha utilitzat el programa 
`plotSocSkillSources.m`, ubicat dins de la carpeta `matlab_programs`. Els resultats s'han guardat a la carpeta `images_brain`.

Per executar el programa Python es realitza de la mateixa manera que els anteriors:
```bash
python calc_common_ICAs_all_mean.py
```

I el programa `plotSocSkillSources.m` s'ha d'executar des de MATLAB.


