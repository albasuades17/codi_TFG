# Projecte d'oculometria

Aquest projecte es basa en carregar, netejar i analitzar les 
dades oculomètriques dels diversos pacients de l'experiment.


## Estructura del projecte
El projecte consta dels següents quatre programes .py:
- carrega_dades.py: carrega les dades de la base de dades i les guarda
a diversos diccionaris.
- neteja_dades.py: neteja i filtra les dades. També es fa la normalització de les dades.
- analisis_dades.py: analitza les dades a partir de diverses regressions lineals.
- json_functions.py: consta de dues funcions que serveixen per guardar i 
carregar diccionaris en format JSON.

També té vàries carpetes amb dades i imatges per poder comprovar que les dades extretes siguin
coherent i també per realitzar l'anàlisi final.


## Requisits
En tot el projecte s'ha usat la versió Python 3.12 i les següents llibreries:
- pandas
- matplolib
- json
- os
- seaborn
- scipy
- sklearn 
- sys
- mysql-connector-python (només és necessari si es volen carregar les dades)

## Fitxers
### 1. carrega_dades.py
**IMPORTANT:** no cal executar aquest programa ja que les dades ja estan computades i guardades.

En el programa es connecta amb la base de dades de MySQL i fa les queries necessàries
per obtenir totes les dades d'interès de l'experiment de cada pacient.

Per tal que hi hagi persistència de dades es crea una carpeta anomenada dades_docs on es guarden
els següent fitxers:
- **tTime.txt:** diccionari on hi ha els temps de mesura de la pupil·la de cada assaig
- **pupilL.txt:** diccionari on hi ha les mesures de la pupil·la de l'esquerra.
- **pupilR.txt:** diccionari on hi ha les mesures de la pupil·la de la dreta.
- **dades_assajos.txt:** diccionari on es guarden les següents dades de cada assaig: SF, SP, 
MedsOn, PD (si és Parkinson o no), tOriginShow i nControlLevel (CT).

En cas que es vulgui provar carregar les dades de la base de dades
s'han de tenir els següents programes amb les versions detallades:
- MySQL Server versió 8.0
- MySQL Workbench versió 8.0

També s'ha de tenir la base de dades carregada al MySQL Workbench.

L'execució del programa es fa de la següent manera (des de terminal):
```bash
python carrega_dades.py "<user_db>" "<password_db>"
```
on <user_db> i <password_db> són l'usuari i la contrasenya, respectivament,
de la connexió feta en el Workbench.

### 2. neteja_dades.py
En aquest fitxer s'han netejat i normalitzat les dades de la pupil·la.

#### 2.1. Neteja de les dades
La neteja de dades s'ha fet individualment per cada pacient seguint els següents passos:
- S'han considerat les dades de cada assaig que estan en el rang  $[tOriginShow - 500, tOriginShow + 1000]$.
Des d'ara anomenarem assaig a les dades que estan en aquest interval.
- En cas de ser una persona sana:
  - Per cada sessió s'ha calculat la mitjana ($mean$) i la desviació estàndard ($std$) dels assajos dels blocs de control
  (bloc 1 i bloc 2). 
- En cas de ser un pacient amb Parkinson:
  - Per cada sessió s'ha calculat la mitjana ($mean$) i la desviació estàndard dels assajos dels blocs de control. 
  $std$ s'ha agafat com la mitjana de les desviacions calculades en cada sessió. 
- **Notació:** Donat una mesura de la pupil·la, denotem que és una mesura bona si es troba en el rang 
$[mean - 2*std, mean + 2*std]$, on $mean$ i $std$ són concrets per cada sessió.
- **Notació:** Donat un assaig, denotem que és un assaig bo si durant l'assaig, per cada 250 ms de temps s'ha
trobat alguna mesura bona en qualsevol dels dos ulls.
- Per tal de guardar les dades en una matriu, s'ha fet que per cada 20 segons d'assaig s'anoti quina mesura bona
s'ha fet per cada ull. En cas que en el mateix interval hi hagi més d'una mesura bona es fa la mitjana, en cada que no
hi hagi cap mesura bona es deixa el valor a null. 
- Els assajos que no són bons s'han eliminat.


#### 2.2. Normalització de les dades
La normalització de les dades s'ha fet de cada pacient i diferenciant les sessions de cada assaig.
- En el cas dels pacients sans:
  - Per cada sessió s'ha calculat la mitjana i desviació estàndard de la manera detallada a l'anterior subapartat. 
  És a dir, s'ha calculat la mitjana ($mean$) i la desviació estàndard ($std$) dels assajos dels blocs de control
  (bloc 1 i bloc 2). Per cada mesura $x$ obtenim la mesura normalitzada $\hat{x}$ aplicant la següent fórmula:
        $\hat{x} = \frac{x - mean}{std}$

- En el cas dels pacients amb Parkinson:
  - Per cada sessió, el pacient està ON o OFF en la medicació. Per tant, denotem
  $x_{ON}$ si és una mesura de la pupil·la quan el pacient estava medicat o $x_{OFF}$ si no ho estava. De la mateixa manera
  denotarem $\mu_{ON}$ i $\sigma_{ON}$ per la mitjana i desviació estàndard quan el pacient estava medicat 
  i $\mu_{OFF}$ i $\sigma_{OFF}$ per la mitjana i desviació estàndard quan el pacient no estava medicat. Tant la mitjana
  com la desviació estàndard es calculen de tots els assajos dels blocs de control (bloc 1 i 2). També es calcula la
  mitjana i desviació estàndard general, és a dir:
    $\mu_{PD} = \frac{\mu_{ON} + \mu_{OFF}}{2}$
    $\sigma_{PD} = \frac{\sigma_{ON} + \sigma_{OFF}}{2}$
  - Finalment, el valor de la pupil·la normalitzada es calcula de la següent manera:
    - Cas en què s'ha calculat en la sessió en què el pacient estava medicat (ON):
    $\hat{x}_{ON} = \frac{x_{ON} - \mu_{ON}}{\sigma_{PD}} + \frac{1}{2} \cdot \frac{\mu_{ON} - \mu_{OFF}}{\mu_{PD}}$
    - Cas en què s'ha calculat en la sessió en què el pacient no estava medicat (OFF):
    $\hat{x}_{OFF} = \frac{x_{OFF} - \mu_{OFF}}{\sigma_{PD}} - \frac{1}{2} \cdot \frac{\mu_{ON} - \mu_{OFF}}{\mu_{PD}}$

Un cop les dades ja estan normalitzades es va fer la mitjana de les pupil·les i la interpolació de les dades que faltaven.
A continuació es van guardar en un fitxer anomenat mitjana_assajos.txt dins de la carpeta dades_docs.

Per tal de poder visualitzar si la neteja i la normalització s'ha fet de manera correcta s'han creat dues carpetes que contenen
diverses gràfiques:
- **img_mean_trial:** conté una imatge per cada ull de cada sessió de cada pacient. En cada imatge hi ha 
sis gràfiques, una per cada bloc realitzat en la sessió. Es dibuixa la mitjana de cada assaig bo realitzat en el bloc.
- **img:** conté una imatge per cada sessió que ha fet cada pacient. En cada imatge hi ha sis files, 
una per cada bloc realitzat en la sessió. En cada fila hi ha quatre columnes: la primera és de les
mesures bones guardades de la pupil·la esquerra, la segona de la pupil·la dreta, la tercera de la combinació
de les dues (la mitjana) i la quarta són els resultats de la pupil·la interpolada. En totes les gràfiques
s'usen les dades de la pupil·la normalitzades.

L'execució del programa es realitza de la següent manera:
```bash
python neteja_dades.py
```

### analisis_dades.py
En aquest programa hi ha quatre mètodes que fan diversos anàlisis de les dades.
Tots es basen en la següent regressió lineal:
- En els pacients de Parkinson:
  $$\phi = \beta_0 + \beta_1 \cdot SF + \beta_2 \cdot SF \times SP + \beta_3 \cdot CT + \beta_4 \cdot MedsON + \beta_5 \cdot MedsOn \times SF + \beta_6 \cdot MedsOn \times SF \times SP$$
- En les persones sanes:
$$\phi = \beta_0 + \beta_1 \cdot SF + \beta_2 \cdot SF \times SP + \beta_3 \cdot CT$$

Sigui $\phi_i$ una observació de la variable aleatòria $\phi$, aleshores $\phi_i$ és la
mitjana de la mesura de la pupil·la normalitzada dels primers 500 ms de l'assaig $i$.

Cal esmentar que en les persones amb Parkinson que només han fet una sessió s'ha
usat el model de la regressió lineal de les persones sanes, ja que les observacions
de MedsOn serien constants.

Un cop trobat els coeficients de la regressió lineal per cada pacient, es calcula el u-test i el 
ks-test de la següent manera: per cada coeficient de la regresió es consideren els valors obtinguts
per cada pacient i es compara amb 0, ja que, en aquest cas, que el coeficient sigui significatiu
és equivalent a demanar que sigui diferent a zero.

En cada un dels mètodes es guarden les següents imatges:
- **barplot_mitjana.png:** barplot de la mitjana dels coeficients de tots els pacients.
- **barplot_pacient.png:** per cada pacient hi ha un barplot que mostra els coeficients obtinguts.
- **regression_results.png:** taula on es mostra per cada pacient quin valor té cada coeficient. 
També es mostra el valor del coeficient de determinació de la regressió lineal considerant tots els pacients.
- **significance_regression.png:** taula on es mostra, per cada coeficient, el valor del u-test, el ks-test i
els p-values obtinguts per cada test.
- **violin_plot.png:** es mostra un violin plot per cada variable.

Tots els resultats obtinguts es guarden a la carpeta results.

A continuació es detalla quines regressions lineals s'usen en cada mètode:
- **regressio_lineal:** es realitza la regressió lineal explicada a dalt. Els resultats es guarden
dins de la carpeta model_total.
- **regressio_lineal_CT:** se separen les dades de la pupil·la segons el 
valor del CT de cada assaig i es fa la següent regressió lineal per cada
grup d'assajos de cada pacient:
  - En els pacients de Parkinson:
    $$\phi = \beta_0 + \beta_1 \cdot SF + \beta_2 \cdot SF \times SP + \beta_4 \cdot MedsON + \beta_5 \cdot MedsOn \times SF + \beta_6 \cdot MedsOn \times SF \times SP$$
  - En les persones sanes:
    $$\phi = \beta_0 + \beta_1 \cdot SF + \beta_2 \cdot SF \times SP$$

  Els resultats es guarden dins de les carpetes model_CT_1 i model_CT_2, on 1
i 2 representen els valors de CT que s'ha agafat per cada model.
- **regressio_lineal_MedsOn:** es fa la regressió lineal per només els pacients
amb Parkinson. De cada pacients se separen els assajos en dos grups, segons si l'assaig
s'ha fet amb medicació (MedsOn = 1) o sense medicació (MedsOn = 0). Es fa la següent regressió lineal:
$$\phi = \beta_0 + \beta_1 \cdot SF + \beta_2 \cdot SF \times SP + \beta_3 \cdot CT$$
Els resultats es guarden dins de les carpetes model_MedsOn_0 i model_MedsOn_1.
- **regressio_lineal_sans:** es fa la regressió lineal només per les persones sanes i es
considera el model de dalt, és a dir:
$$\phi = \beta_0 + \beta_1 \cdot SF + \beta_2 \cdot SF \times SP + \beta_3 \cdot CT$$

  
L'execució del programa es fa de la següent manera:
```bash
python analisis_dades.py
```