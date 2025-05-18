TARGET		= Cap
DESTDIR		= bin
MOC_DIR		= bin
OBJECTS_DIR	= bin
QT		+= core network widgets
RC_ICONS	= img/flow.ico

#DEFINES += QT_DEPRECATED_WARNINGS
HEADERS += lib.h
SOURCES += app.cpp main.cpp

QMAKE_CXXFLAGS += -std=c++20

CONFIG -= debug debug_and_release release

