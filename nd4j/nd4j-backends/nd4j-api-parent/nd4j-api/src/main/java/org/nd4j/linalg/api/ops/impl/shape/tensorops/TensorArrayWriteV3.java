package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.list.compat.TensorList;

public class TensorArrayWriteV3 extends BaseTensorOp {

   public TensorArrayWriteV3(String name, SameDiff sameDiff, SDVariable[] args){
      super(name, sameDiff, args);
   }
   public TensorArrayWriteV3(SameDiff sameDiff, SDVariable[] args){
      super(null, sameDiff, args);
   }

   public TensorArrayWriteV3(){}
   @Override
   public String tensorflowName() {
      return "TensorArrayWriteV3";
   }

   @Override
   public TensorList execute(SameDiff sameDiff) {
      val list = getList(sameDiff);

      val ids =getArgumentArray(1).getInt(0);
      val array = getArgumentArray(2);

      list.put(ids, array);

      return list;
   }

   @Override
   public String opName() {
      return "tensorarraywritev3";
   }

   @Override
   public Op.Type opType() {
      return Op.Type.CUSTOM;
   }
}
