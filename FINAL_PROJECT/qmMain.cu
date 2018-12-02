/*
 * @Author: chaomy
 * @Date:   2018-12-01 21:01:43
 * @Last Modified by:   chaomy
 * @Last Modified time: 2018-12-02 00:24:36
 */

#include "qmHome.h"

int main() {
  QMParallel qm;
  qm.readTrueTable("input.pla"); 
  qm.runQM(128); 
  return (0);
}