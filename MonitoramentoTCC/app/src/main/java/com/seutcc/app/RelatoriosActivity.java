package com.seutcc.app;

import android.graphics.Color;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.formatter.ValueFormatter;
import com.seutcc.app.network.RetrofitClient;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class RelatoriosActivity extends AppCompatActivity {

    private BarChart chart;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_relatorios);

        chart = findViewById(R.id.chartReports);
        configurarGrafico();
        carregarDados();
    }

    private void configurarGrafico() {
        chart.getDescription().setEnabled(false);
        chart.setDrawGridBackground(false);

        XAxis xAxis = chart.getXAxis();
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);
        xAxis.setDrawGridLines(false);
        xAxis.setGranularity(1f); // Mostrar todos os números inteiros
        xAxis.setValueFormatter(new ValueFormatter() {
            @Override
            public String getFormattedValue(float value) {
                return ((int) value) + "h"; // Ex: "10h"
            }
        });
    }

    private void carregarDados() {
        // Chama a porta 8003 (Reports)
        RetrofitClient.getReportService().getStats().enqueue(new Callback<Map<String, Integer>>() {
            @Override
            public void onResponse(Call<Map<String, Integer>> call, Response<Map<String, Integer>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    plotarDados(response.body());
                }
            }
            @Override
            public void onFailure(Call<Map<String, Integer>> call, Throwable t) {}
        });
    }

    private void plotarDados(Map<String, Integer> dados) {
        List<BarEntry> entries = new ArrayList<>();

        // Converte o Map JSON (ex: {"10": 5}) para Entradas do Gráfico
        for (Map.Entry<String, Integer> entry : dados.entrySet()) {
            try {
                float hora = Float.parseFloat(entry.getKey());
                float qtd = entry.getValue();
                entries.add(new BarEntry(hora, qtd));
            } catch (Exception e) {}
        }

        BarDataSet set = new BarDataSet(entries, "Ocorrências");
        set.setColor(Color.BLUE);
        set.setValueTextSize(14f);

        BarData data = new BarData(set);
        data.setBarWidth(0.9f);

        chart.setData(data);
        chart.invalidate(); // Redesenha o gráfico
        chart.animateY(1500); // Animação
    }
}