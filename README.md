# **Soft Actor-Critic (SAC)**
Bienvenido al repositorio de Soft Actor-Critic (SAC), un algoritmo de aprendizaje por refuerzo para resolver problemas de control óptimo. El objetivo de este repositorio es proporcionar una implementación de referencia de SAC y brindar una guía para ayudar a los usuarios a entender y utilizar el algoritmo.

## **¿Qué es Soft Actor-Critic?**
Soft Actor-Critic (SAC) es un algoritmo de aprendizaje por refuerzo desarrollado por Haarnoja et al. en 2018. Es una variante de los algoritmos Actor-Critic, que combinan las ideas de los algoritmos de política (Actor) y valor (Critic) para resolver problemas de control óptimo. A diferencia de otros algoritmos Actor-Critic, SAC utiliza una función objetivo suave para mejorar el rendimiento en tareas con estados continuos y acciones continuas.

## **¿Cómo funciona?**
SAC se basa en tres componentes principales: el actor, el crítico y el valor. El actor es responsable de generar acciones en función del estado actual, el crítico es responsable de evaluar la calidad de las acciones y el valor es responsable de evaluar el valor esperado de un estado.

SAC utiliza dos redes de críticos para calcular el valor esperado de una acción, una función objetivo suave para calcular la política y una red de valor para calcular el valor esperado de un estado. El algoritmo también utiliza una técnica de repetición de experiencias para mejorar el rendimiento.

El entrenamiento se realiza mediante el uso de gradiente de descenso estocástico para actualizar las redes de actor, crítico y valor.

## **¿Por qué usar Soft Actor-Critic?**
SAC tiene varias ventajas sobre otros algoritmos de aprendizaje por refuerzo:

- Puede manejar tareas con estados y acciones continuas de manera efectiva.
- Utiliza una función objetivo suave para mejorar el rendimiento en tareas con estados continuos y acciones continuas.
- Es capaz de aprender políticas estocásticas para mejorar la estabilidad y la exploración.
- Utiliza dos redes de críticos para mejorar la estabilidad del entrenamiento.
- Utiliza una técnica de repetición de experiencias para mejorar el rendimiento.


Este repositorio proporciona una implementación eficiente de Soft Actor-Critic en Tensorflow 2, para resolver problemas de aprendizaje por refuerzo continuo. Incluye una clase SAC que contiene los métodos para entrenar y evaluar el modelo, así como un ejemplo de uso para un ambiente de prueba en MUJOCO.