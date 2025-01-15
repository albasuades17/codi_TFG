# El model
Aquest projecte es basa en l'ús del model implementat a la carpeta `pyMOU` per generar senyals que expliquin els canvis motivacionals durant l'experiment.

## Estructura del projecte
El projecte consta de tres carpetes principals:
- **generated_signals**: Carpeta on es guarden els senyals generats per un participant amb una certa banda de freqüència, un cert moviment i un cert estat motivacional (SP o SF). Els fitxers tenen el següent format de nom: `generated_rand_signals_<motiv>_<num_band>_<num_sub>_<num_motion>`.
- **pyMOU**: Carpeta que conté els models MOU implementats per David de la Osa.
- **results**: Carpeta on es guarden les precisions, les matrius de confusió del classificador i les imatges resultants de la comparació entre els senyals produïts pel model i els senyals reals.

## Requisits
- **Python**: Versió 3.12.

## Explicació del codi
Tal com s'ha esmentat, aquest codi es basa, en bona part, en el TFG de David de la Osa. Per tant, no s'explicarà el contingut que ell ha implementat (fitxers dins la carpeta `pyMOU`). No obstant això, cal esmentar que el notebook `MOU_with_EEGv3.ipynb`, originalment implementat per ell, ha estat modificat per adaptar-se a les dades utilitzades en aquest projecte. Aquest mòdul serveix per visualitzar els senyals disponibles i com el model els processa, permetent una avaluació preliminar de la seva idoneïtat.

Ara, es descriuran els dos programes desenvolupats per aquest projecte.

### `generator_signals.py`
Aquest programa genera senyals a partir dels \textit{envelopes} creats pel model. Itera per les tres bandes de freqüència (`alpha`, `beta` i `gamma`), per tots els participants sans, i pels dos tipus de moviment (`Stop-In` i `Cross-Over`). A més, diferencia els dos tipus de motivació: `SF` (Facilitació Social) i `SP` (Pressió Social).  
Per a cada combinació, guarda els senyals simulats obtinguts a la carpeta `generated_signals`.

Aquest programa conté la següent funció:
- **`get_envelopes`**: Rep la mitjana dels assajos per a un valor determinat d'un estat motivacional, com ara la mitjana dels assajos quan `SF=0`. A partir de la transformada de Hilbert, obté l'\textit{envolvent}.

Per executar el programa només cal introduir a terminal
```bash
python generator_signals.py
```
però es recomana no fer-ho ja que l'execució d'aquest és molt lenta.

### `classifier_signals.py`
Aquest programa conté dues funcions principals per classificar els senyals i, posteriorment, dibuixar els resultats amb l'objectiu de facilitar la comparació. A continuació, es detallen les funcions:
- **`classification_envelope_signals`**: Classifica els senyals simulats emmagatzemats mitjançant el classificador MLR. Posteriorment, guarda les precisions i les matrius de confusió obtingudes a la carpeta `results`.
- **`classification_real_signals`**: Classifica els senyals dels EEG, seguint la metodologia prèvia del projecte d'anàlisi electroencefalogràfica. També guarda les precisions i les matrius de confusió obtingudes a la carpeta `results`.

Finalment, per a cada tipus de motivació, el programa genera un \textit{violin plot} i una matriu de confusió. Els resultats es guarden a la carpeta `results`.

Per executar el programa només cal introduir a terminal
```bash
python classifier_signals.py
```



