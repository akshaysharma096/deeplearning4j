/*
 * Copyright 2016 Skymind
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.deeplearning4j.scalnet.layers.noise

import org.deeplearning4j.nn.conf.dropout.{ AlphaDropout => JAlphaDropout }
import org.deeplearning4j.nn.conf.layers.DropoutLayer
import org.deeplearning4j.scalnet.layers.core.Layer

/**
  * AlphaDropout layer
  *
  * @author Max Pumperla
  */
class AlphaDropout(nOut: List[Int], nIn: List[Int], rate: Double, override val name: String) extends Layer {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new DropoutLayer.Builder()
      .dropOut(new JAlphaDropout(rate))
      .nIn(inputShape.last)
      .nOut(outputShape.last)
      .name(name)
      .build()

  override val outputShape: List[Int] = nOut

  override val inputShape: List[Int] = nIn

  override def reshapeInput(newIn: List[Int]): AlphaDropout =
    new AlphaDropout(nOut, newIn, rate, name)
}

object AlphaDropout {
  def apply(nOut: Int, nIn: Int = 0, rate: Double, name: String = ""): AlphaDropout =
    new AlphaDropout(List(nOut), List(nIn), rate, name)
}
