import { useState, useEffect, useCallback } from 'react';

export function useGridAssets(table: string = 'egat_power_plants', limit: number = 100) {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = useCallback(async () => {
        try {
            setLoading(true);
            const response = await fetch(`/api/gridtokenx/assets?table=${table}&limit=${limit}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${table}: ${response.statusText}`);
            }
            const json = await response.json();
            setData(json);
            setError(null);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [table, limit]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    return { data, loading, error, refresh: fetchData };
}
