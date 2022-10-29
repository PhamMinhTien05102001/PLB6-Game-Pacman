import { observer } from 'mobx-react-lite';
import { useGame } from '../../../components/StoreContext';
import './Progress.css';
import classNames from 'classnames';
import ProgressBar from '@ramonak/react-progress-bar';

export const Progress = observer<{ className?: string }>(({ className }) => {
  const store = useGame();
  return (
    <div className={classNames('Progress', className)}>
      <span>Power</span>
      <ProgressBar
        completed={Math.floor(store.power)}
        className="progress-bar"
        bgColor="#2121FF"
      />
    </div>
  );
});
