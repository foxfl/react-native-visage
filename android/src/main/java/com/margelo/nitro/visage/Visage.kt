package com.margelo.nitro.visage
  
import com.facebook.proguard.annotations.DoNotStrip

@DoNotStrip
class Visage : HybridVisageSpec() {
  override fun multiply(a: Double, b: Double): Double {
    return a * b
  }
}
