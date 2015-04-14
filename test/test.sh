#!/bin/bash

th -loxnn -e "oxnn.test()"
th -loxnn -e "oxnn.InitCuda() oxnn.test()"

