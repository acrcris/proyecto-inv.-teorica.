### Que ID y OOD en inteligencia artificial

- ID In-Distribution (“dentro de la distribución”)
Son los datos similares en tipo, generación, dominio o distribución estadística a los que se usaron para entrenar el modelo. Si el modelo fue entrenado con datos del tipo X, los datos ID son datos como los X: mismos generadores, mismas condiciones, mismos “supuestos”.

- OOD = Out-Of-Distribution (“fuera de la distribución”)
Son los datos que no se parecen a los de entrenamiento: pueden venir de otra fuente, presentar variaciones que no se vieron antes (diferente distribución de entrada), tener clases nuevas, otros niveles de ruido, contextos distintos, etc. En otras palabras: datos cuya distribución difiere de la distribución ID.



        La idea en este paso es revisar el comportamiento del modelo en datos dentro de la distribucion, pero entonces cuales son datos dentro de la distribucion segun la librerias o se lo que sea que este haciendo el codigo?

        En este caso los datos ID son los datos de entrenamiento SATNet

        Cuales son los datos OOD?

        En este caso los datos OOD son los datos de RNN

La es reproducir los resultado del  ≥99% ID y ≥70% OOD con test-time extension (T=128) y energy voting (K≈100)


### Que es ITrSA y transformers