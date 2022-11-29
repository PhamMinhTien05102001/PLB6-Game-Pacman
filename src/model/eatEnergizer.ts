import { Game } from './Game';
import { action } from 'mobx';

export const ENERGIZER_POINTS = 10;

export const eatEnergizer = action((game: Game) => {
  game.score += ENERGIZER_POINTS;
  game.pillsCount += 1;
  game.killedGhosts = 0;
  game.pacMan.send('ENERGIZER_EATEN');
  for (const ghost of game.ghosts) {
    ghost.send('ENERGIZER_EATEN');
  }
});
