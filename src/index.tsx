import { NitroModules } from 'react-native-nitro-modules';
import type { Visage } from './Visage.nitro';

const VisageHybridObject =
  NitroModules.createHybridObject<Visage>('Visage');

export function multiply(a: number, b: number): number {
  return VisageHybridObject.multiply(a, b);
}
