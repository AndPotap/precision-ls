########################################################
# Sanity checks for data generation and task sampling.
########################################################

# Primitives
python -m src.datagen.main.primitives

# LeastSquaresICL
python -m src.datagen.main.least_squares_icl

# LeastSquares
python -m src.datagen.main.least_squares

# ExplicitGradient
python -m src.datagen.main.explicit_gradient

# MultistepGradientDescent
python -m src.datagen.main.multistep_gd

# ODEs
python -m src.datagen.main.odes