import React from 'react';

interface ApiHookState<T> {
  state: T;
  loading: boolean;
}

export function useAPI(
  endpoint: string,
  method: 'GET' | 'POST',
  data?: any
): ApiHookState<string> {
  const [state, setState] = React.useState<string>('');
  const [loading, setLoading] = React.useState<boolean>(true);
  const options: { method: 'GET' | 'POST'; body: any } = {
    method,
    body: data || '',
  };
  if (options.method === 'GET') delete options.body;

  React.useEffect((): void => {
    fetch(endpoint, options)
      .then((response: Response): void => {
        if (response.status === 200) {
          response.json().then((data: { title: string }) => {
            setState(data.title);
            setLoading(false);
          });
        }
        setLoading(false);
      })
      .catch((err: Error) => {
        setLoading(false);
      });
  }, [endpoint]);

  return {
    state,
    loading,
  };
}
