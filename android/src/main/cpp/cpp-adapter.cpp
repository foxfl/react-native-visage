#include <jni.h>
#include "visageOnLoad.hpp"

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return margelo::nitro::visage::initialize(vm);
}
