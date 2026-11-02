import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  SafeAreaView,
  StatusBar,
  Alert,
} from 'react-native';

interface ForecastPoint {
  month_index: number;
  target_date: string;
  predicted_price: number;
  lower_bound_95: number;
  upper_bound_95: number;
}

export default function Way2AgriMobileScreen() {
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [forecasts, setForecasts] = useState<ForecastPoint[]>([]);
  const [latestPrice, setLatestPrice] = useState<number>(15750);
  const [modelName, setModelName] = useState<string>('Hybrid Ensemble');

  // FastAPI or Django endpoint (e.g. 10.0.2.2 for Android emulator or LAN IP for device)
  const API_URL = 'http://10.0.2.2:8001/predict/forecast-12m';

  const fetchForecast = async () => {
    try {
      const response = await fetch(API_URL);
      if (!response.ok) throw new Error('Network error');
      const data = await response.json();
      setForecasts(data.forecast_points || []);
      setLatestPrice(data.latest_historical_price || 15750);
      setModelName(data.model_name || 'Hybrid Ensemble');
    } catch (error) {
      console.warn('Falling back to local cache or offline storage:', error);
      // Fallback sample data for offline resiliency
      setForecasts([
        { month_index: 1, target_date: '2025-10-01', predicted_price: 15771.46, lower_bound_95: 13770.77, upper_bound_95: 17772.15 },
        { month_index: 2, target_date: '2025-11-01', predicted_price: 15786.11, lower_bound_95: 13706.94, upper_bound_95: 17865.28 },
        { month_index: 3, target_date: '2025-12-01', predicted_price: 15430.61, lower_bound_95: 13275.81, upper_bound_95: 17585.41 },
        { month_index: 4, target_date: '2026-01-01', predicted_price: 15271.50, lower_bound_95: 13043.63, upper_bound_95: 17499.37 },
        { month_index: 5, target_date: '2026-02-01', predicted_price: 14442.23, lower_bound_95: 12143.62, upper_bound_95: 16740.84 },
      ]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchForecast();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchForecast();
  };

  const handleActionAlert = (item: ForecastPoint) => {
    if (item.predicted_price < 15000) {
      Alert.alert(
        'AI Procurement Recommendation',
        `Price drops to ₹${item.predicted_price.toFixed(2)} in ${item.target_date}.\nRecommended Action: Lock in forward contracts with farmer producer organizations (FPOs) now to capture lowest seasonal cost!`,
        [{ text: 'Acknowledge', style: 'default' }]
      );
    } else {
      Alert.alert(
        'Forecast Details',
        `Target Date: ${item.target_date}\nExpected Price: ₹${item.predicted_price.toFixed(2)}\nRange: ₹${item.lower_bound_95.toFixed(2)} - ₹${item.upper_bound_95.toFixed(2)}`,
        [{ text: 'Close', style: 'cancel' }]
      );
    }
  };

  const renderItem = ({ item }: { item: ForecastPoint }) => {
    const isDip = item.predicted_price < 15000;
    const isPeak = item.predicted_price > 16100;

    return (
      <TouchableOpacity
        style={[styles.card, isDip && styles.dipCard, isPeak && styles.peakCard]}
        onPress={() => handleActionAlert(item)}
        activeOpacity={0.7}
      >
        <View style={styles.cardHeader}>
          <Text style={styles.monthBadge}>Month +{item.month_index}</Text>
          <Text style={styles.dateText}>{item.target_date}</Text>
        </View>

        <View style={styles.priceRow}>
          <Text style={styles.priceLabel}>Predicted Price:</Text>
          <Text style={styles.priceValue}>
            ₹{item.predicted_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </Text>
        </View>

        <View style={styles.ciRow}>
          <Text style={styles.ciText}>
            95% Range: ₹{Math.round(item.lower_bound_95)} - ₹{Math.round(item.upper_bound_95)}
          </Text>
          {isDip && <Text style={styles.buySignal}>BUY DIP</Text>}
          {isPeak && <Text style={styles.peakSignal}>PEAK</Text>}
        </View>
      </TouchableOpacity>
    );
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#059669" />
        <Text style={styles.loadingText}>Fetching Way2Agri Forecasts...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f8fafc" />
      <View style={styles.header}>
        <Text style={styles.orgTag}>Way2Agribusiness India Pvt Ltd</Text>
        <Text style={styles.title}>Field Price Forecasts</Text>
        <View style={styles.summaryBar}>
          <View>
            <Text style={styles.summarySub}>Spot Price</Text>
            <Text style={styles.summaryValue}>₹{latestPrice.toLocaleString()}</Text>
          </View>
          <View>
            <Text style={styles.summarySub}>Engine</Text>
            <Text style={styles.summaryValue}>{modelName}</Text>
          </View>
        </View>
      </View>

      <FlatList
        data={forecasts}
        keyExtractor={(item) => item.target_date}
        renderItem={renderItem}
        contentContainerStyle={styles.list}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={['#059669']} />}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8fafc' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  loadingText: { marginTop: 12, color: '#065f46', fontWeight: '600' },
  header: { padding: 16, backgroundColor: '#ffffff', borderBottomWidth: 1, borderBottomColor: '#e2e8f0' },
  orgTag: { fontSize: 11, fontWeight: '700', color: '#059669', textTransform: 'uppercase' },
  title: { fontSize: 22, fontWeight: '800', color: '#0f172a', marginTop: 2 },
  summaryBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#f1f5f9',
    padding: 12,
    borderRadius: 8,
    marginTop: 12,
  },
  summarySub: { fontSize: 11, color: '#64748b' },
  summaryValue: { fontSize: 15, fontWeight: '700', color: '#0f172a', marginTop: 2 },
  list: { padding: 16 },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  dipCard: { borderColor: '#10b981', borderLeftWidth: 5 },
  peakCard: { borderColor: '#f43f5e', borderLeftWidth: 5 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  monthBadge: { fontSize: 12, fontWeight: '700', color: '#059669' },
  dateText: { fontSize: 13, color: '#64748b', fontWeight: '500' },
  priceRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 },
  priceLabel: { fontSize: 13, color: '#475569' },
  priceValue: { fontSize: 18, fontWeight: '800', color: '#0f172a' },
  ciRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 },
  ciText: { fontSize: 11, color: '#94a3b8' },
  buySignal: { fontSize: 10, fontWeight: '800', color: '#059669', backgroundColor: '#ecfdf5', paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4 },
  peakSignal: { fontSize: 10, fontWeight: '800', color: '#e11d48', backgroundColor: '#fff1f2', paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4 },
});
