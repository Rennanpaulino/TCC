package com.seutcc.app;

import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;
import com.seutcc.app.models.Previsao;
import com.seutcc.app.models.Report;
import com.seutcc.app.network.RetrofitClient;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DetalhesEstacaoActivity extends AppCompatActivity {

    // Componentes de Texto e Loading
    private TextView txtNomeEstacao, txtTimer, txtAviso;
    private ProgressBar loading;

    // Componentes da Lotação
    private TextView txtQtdPessoas, txtTituloLotacao;
    private CardView cardLotacao;

    // Botão de Ação
    private ExtendedFloatingActionButton fabReport;

    // Dados da Navegação
    private String linhaId, estacaoId, nomeEstacao;

    // Controlador do Loop
    private Handler handler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detalhes_estacao);

        // 1. Receber dados da tela anterior
        linhaId = getIntent().getStringExtra("LINHA_ID");
        estacaoId = getIntent().getStringExtra("ESTACAO_ID");
        nomeEstacao = getIntent().getStringExtra("ESTACAO_NOME");

        // 2. Vincular componentes do XML (IDs)
        txtNomeEstacao = findViewById(R.id.txtNomeEstacao);
        txtTimer = findViewById(R.id.txtTimer);
        loading = findViewById(R.id.loading);
        txtAviso = findViewById(R.id.txtAviso);
        fabReport = findViewById(R.id.fabReport);

        // IDs do Cartão de Lotação
        cardLotacao = findViewById(R.id.cardLotacao);
        txtTituloLotacao = findViewById(R.id.txtTituloLotacao);
        txtQtdPessoas = findViewById(R.id.txtQtdPessoas);

        // 3. Configuração Inicial
        txtNomeEstacao.setText(nomeEstacao);

        // Clique do Botão de Reportar
        fabReport.setOnClickListener(v -> mostrarDialogoReport());

        // 4. Iniciar Loop
        iniciarLoop();
    }

    // --- LOOP DE ATUALIZAÇÃO ---

    private void iniciarLoop() {
        handler.post(new Runnable() {
            @Override
            public void run() {
                buscarPrevisao();
                handler.postDelayed(this, 5000); // Roda a cada 5 segundos
            }
        });
    }

    private void buscarPrevisao() {
        RetrofitClient.getIoTService().getPrevisao(linhaId, estacaoId).enqueue(new Callback<Previsao>() {
            @Override
            public void onResponse(Call<Previsao> call, Response<Previsao> response) {
                loading.setVisibility(View.GONE); // Esconde o loading girando

                if (response.isSuccessful() && response.body() != null) {
                    Previsao p = response.body();
                    atualizarTela(p);
                }
            }

            @Override
            public void onFailure(Call<Previsao> call, Throwable t) {
                loading.setVisibility(View.GONE);
            }
        });
    }

    private void atualizarTela(Previsao p) {
        // A. Atualiza Timer
        if (p.getMensagem() != null && p.getMensagem().contains("Nenhum trem")) {
            txtTimer.setText("S/ Trem");
            txtTimer.setTextSize(40);
        } else {
            String tempo = String.format("%02d : %02d", p.getMinutos(), p.getSegundos());
            txtTimer.setText(tempo);
            txtTimer.setTextSize(64);
        }

        // B. Atualiza Aviso de GPS
        if (p.isEstimado()) {
            txtAviso.setVisibility(View.VISIBLE);
            txtAviso.setText("⚠️ Sinal GPS Instável (Estimado)");
        } else {
            txtAviso.setVisibility(View.GONE);
        }

        // C. ATUALIZA CARTÃO DE LOTAÇÃO (Lógica Visual)
        int qtd = p.getLotacao(); // O Backend precisa mandar esse campo "lotacao" no JSON de previsão
        txtQtdPessoas.setText(qtd + " pessoas detectadas");

        if (qtd < 10) {
            // VERDE (Vazio)
            cardLotacao.setCardBackgroundColor(Color.parseColor("#E8F5E9"));
            txtTituloLotacao.setText("VAGÃO VAZIO");
            txtTituloLotacao.setTextColor(Color.parseColor("#2E7D32"));
            txtQtdPessoas.setTextColor(Color.parseColor("#2E7D32"));
        } else if (qtd < 30) {
            // LARANJA (Moderado)
            cardLotacao.setCardBackgroundColor(Color.parseColor("#FFF3E0"));
            txtTituloLotacao.setText("LOTAÇÃO MÉDIA");
            txtTituloLotacao.setTextColor(Color.parseColor("#EF6C00"));
            txtQtdPessoas.setTextColor(Color.parseColor("#EF6C00"));
        } else {
            // VERMELHO (Cheio)
            cardLotacao.setCardBackgroundColor(Color.parseColor("#FFEBEE"));
            txtTituloLotacao.setText("SUPERLOTADO");
            txtTituloLotacao.setTextColor(Color.parseColor("#C62828"));
            txtQtdPessoas.setTextColor(Color.parseColor("#C62828"));
        }
    }

    // --- SISTEMA DE REPORT ---

    private void mostrarDialogoReport() {
        String[] problemas = {
                "Trem muito atrasado",
                "Superlotação Extrema",
                "Segurança / Furto",
                "Sujeira / Limpeza",
                "Ar Condicionado Quebrado",
                "Outros"
        };

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Qual o problema?");

        builder.setItems(problemas, (dialog, which) -> {
            enviarReportBackend(problemas[which]);
        });

        builder.setNegativeButton("Cancelar", null);
        builder.show();
    }

    private void enviarReportBackend(String problema) {
        // Cria o Report
        Report report = new Report("Passageiro", linhaId, estacaoId, problema);}}