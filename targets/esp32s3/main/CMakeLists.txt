# The generated detector is ordinary C99 and needs no component dependencies.
idf_component_register(
    SRCS "main.c" "generated/kangdn.c"
    INCLUDE_DIRS "." "generated"
)
