import sys
import mysql.connector
from json_functions import save_data

# Diccionaris per tal de guardar les dades de la base de dades
tTime = dict()
pupilL = dict()
pupilR = dict()
dades_assajos = dict()

# Credencials per connectar-se amb la base de dades
user_db = str(sys.argv[1])
pass_db = str(sys.argv[2])

# Funció que fa les queries necessàries per obtenir les dades d'interés de la base de dades
def get_data(cnx):
    cursor = cnx.cursor()
    # Hem d'estudiar els pacients del 25 al 35 (pacients sans)
    # i del 53 al 70 (pacients amb Parkinson)
    parkinson_desease = False
    pacients_nums = [i for i in range (25, 36)]
    for i in range (53, 71):
        pacients_nums.append(i)

    for pacient in pacients_nums:
        if pacient > 36:
            parkinson_desease = True

        tTime[pacient] = dict()
        pupilL[pacient] = dict()
        pupilR[pacient] = dict()
        dades_assajos[pacient] = dict()

        #Cada pacient pot haver fet dues sessions, que tenen com identificador 1 o 2
        for idSessio in [1,2]:
            # Veiem quines sessions ha fet cada pacient
            cursor.execute("""
                SELECT DISTINCT(nBlockOrderS%s) FROM _trials WHERE sessionID = %s AND idSubject = %s;
                """,
                           (idSessio, idSessio, pacient)
                           )
            result = cursor.fetchall()
            # Només considerem els pacients que hagin fet algun assaig
            if len(result) > 0:
                tTime[pacient][idSessio] = dict()
                pupilL[pacient][idSessio] = dict()
                pupilR[pacient][idSessio] = dict()
                dades_assajos[pacient][idSessio] = dict()

                # Ordenem segons l'atribut idTrial per saber en quin ordre s'han fet els blocs.
                #Usem aquesta variable perquè en alguns casos, nBlockOrderS1 i nBlockOrderS2 tenen valor null
                cursor.execute("""
                                SELECT T.nBlock, T.nTrial, O.tTime, O.pupilsizeL, O.pupilsizeR, 
                                T.tOriginShow, T.MedsOn, T.nMotivationLevel, T.nMajorMinor, T.nChoice, T.nControlLevel 
                                FROM oculometry O INNER JOIN _trials T ON O.idTrial = T.idTrial 
                                WHERE T.idSubject = %s AND T.sessionID = %s ORDER BY T.idTrial, T.nTrial, O.tTime;
                                """,
                               (pacient, idSessio)
                               )
                result = cursor.fetchall()
                if len(result) == 0:
                    del tTime[pacient]
                    del pupilL[pacient]
                    del pupilR[pacient]

                bloc = 0
                assaig = 0
                delete_assaig = False
                # Iterem per les files de la query executada
                for row in result:
                    # Mirem si hem canviat de bloc
                    if row[0] != bloc:
                        bloc = row[0]
                        assaig = row[1]

                        # Ara, com a keys dins d'una sessió tindrem els blocs ordenats que ha fet el pacient
                        tTime[pacient][idSessio][bloc] = dict()
                        pupilL[pacient][idSessio][bloc] = dict()
                        pupilR[pacient][idSessio][bloc] = dict()
                        dades_assajos[pacient][idSessio][bloc] = dict()

                        # Mirem que la variable nControlLever no sigui nul·la
                        # Si no, eliminem l'assaig
                        if row[10] == None:
                            delete_assaig = True
                            dades_assajos[pacient][idSessio][bloc][assaig] = {'Eliminat': True}
                        else:
                            delete_assaig = False
                            # Ara veiem quin valor prenen les variables SF (social facilitation) i SP (social pressure)
                            if int(row[7]) == 1 or int(row[7]) == 2:
                                sf = 1
                                # Farem que el valor de social pressure sigui de 0 o 1
                                sp = int(row[7]) - 1
                            else:
                                sf = 0
                                sp = 0
                            # Només considerem l'atribut medsOn si el pacient té Parkinson
                            if parkinson_desease:
                                meds_on = row[6]
                            else:
                                meds_on = 0

                            dades_assajos[pacient][idSessio][bloc][assaig] = {'tOriginShow': row[5], 'PD': parkinson_desease,
                                                                              'MedsOn': meds_on, 'SF': sf, 'SP': sp,
                                                                              'nMajorMinor': row[8],'nControlLevel': row[10],
                                                                              'Eliminat': False}
                            tTime[pacient][idSessio][bloc][assaig] = list()
                            pupilL[pacient][idSessio][bloc][assaig] = list()
                            pupilR[pacient][idSessio][bloc][assaig] = list()

                    # Mirem si hem canviat d'assaig
                    if assaig != row[1]:
                        assaig = row[1]
                        # Mirem que la variable nControlLever no sigui nul·la
                        if row[10] != None:
                            delete_assaig = False
                            tTime[pacient][idSessio][bloc][assaig] = list()
                            pupilL[pacient][idSessio][bloc][assaig] = list()
                            pupilR[pacient][idSessio][bloc][assaig] = list()

                            # Ara veiem quin valor prenen les variables SF (social facilitation) i SP (social pressure)
                            if int(row[7]) == 1 or int(row[7]) == 2:
                                sf = 1
                                sp = int(row[7]) - 1
                            else:
                                sf = 0
                                sp = 0
                            # Només considerem l'atribut medsOn si el pacient té Parkinson
                            if parkinson_desease:
                                meds_on = row[6]
                            else:
                                meds_on = 0
                            dades_assajos[pacient][idSessio][bloc][assaig] = {'tOriginShow': row[5], 'PD': parkinson_desease, 'MedsOn': meds_on,
                                                                              'SF': sf, 'SP': sp,
                                                                              'nMajorMinor': row[8], 'nControlLevel': row[10],
                                                                              'Eliminat': False}
                        else:
                            dades_assajos[pacient][idSessio][bloc][assaig] = {'Eliminat': True}
                            delete_assaig = True

                    if not delete_assaig:
                        # Introduïm les dades trobades en la query
                        tTime[pacient][idSessio][bloc][assaig].append(row[2])
                        pupilL[pacient][idSessio][bloc][assaig].append(row[3])
                        pupilR[pacient][idSessio][bloc][assaig].append(row[4])

    cursor.close()


# Ens connectem amb la base de dades
cnx = mysql.connector.connect(user=user_db, password=pass_db, host="127.0.0.1", database="tfg_db")

get_data(cnx)

# Guardem totes les dades obtingudes en fitxers .txt
save_data('dades_docs/' + "tTime", tTime)
save_data('dades_docs/' + "pupilL", pupilL)
save_data('dades_docs/' + "pupilR", pupilR)
save_data('dades_docs/' + "dades_assajos", dades_assajos)

cnx.close()


