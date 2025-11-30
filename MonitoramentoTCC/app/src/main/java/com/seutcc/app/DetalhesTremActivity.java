package com.seutcc.app;

import android.graphics.Color;
import android.location.Location;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.PolylineOptions;
import com.seutcc.app.models.Trem;
import com.seutcc.app.network.RetrofitClient;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DetalhesTremActivity extends AppCompatActivity implements OnMapReadyCallback {

    private GoogleMap mMap;
    private String tremId;
    private TextView txtEta, txtAviso;
    private Handler handler = new Handler();

    // Coordenada Fixa de Santo Amaro (TCC)
    private final LatLng DESTINO = new LatLng(-23.65637, -46.70956);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detalhes_trem);

        tremId = getIntent().getStringExtra("ID_TREM");

        TextView txtTitulo = findViewById(R.id.txtTituloTrem);
        txtTitulo.setText("Monitorando: " + tremId);

        txtEta = findViewById(R.id.txtEta);
        txtAviso = findViewById(R.id.txtAvisoGps);

        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager().findFragmentById(R.id.map);
        mapFragment.getMapAsync(this);
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        mMap = googleMap;

        // Adiciona Marcador FIXO da Estação
        mMap.addMarker(new MarkerOptions()
                .position(DESTINO)
                .title("Estação Santo Amaro")
                .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_AZURE)));

        atualizarPosicao();
    }

    private void atualizarPosicao() {
        RetrofitClient.getIoTService().getTrens().enqueue(new Callback<List<Trem>>() {
            @Override
            public void onResponse(Call<List<Trem>> call, Response<List<Trem>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    // Procura o trem específico na lista
                    for (Trem t : response.body()) {
                        if (t.getId().equals(tremId)) {
                            plotarNoMapa(t);
                            break;
                        }
                    }
                }
            }
            @Override
            public void onFailure(Call<List<Trem>> call, Throwable t) {}
        });

        // Atualiza a cada 3s
        handler.postDelayed(this::atualizarPosicao, 3000);
    }

    private void plotarNoMapa(Trem t) {
        LatLng pos = new LatLng(t.getLatitude(), t.getLongitude());

        // Limpa marcadores anteriores (exceto a estação se quiser lógica complexa, mas clear() limpa tudo)
        mMap.clear();

        // Recria Estação
        mMap.addMarker(new MarkerOptions().position(DESTINO).title("Destino: Sto Amaro").icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_BLUE)));

        // Cria Marcador do Trem
        mMap.addMarker(new MarkerOptions()
                .position(pos)
                .title(t.getId())
                .snippet("Velocidade: " + t.getVelocidade() + " km/h"));

        // Desenha Linha entre Trem e Estação
        mMap.addPolyline(new PolylineOptions().add(pos, DESTINO).width(5).color(Color.GRAY));

        // Move a camera
        mMap.animateCamera(CameraUpdateFactory.newLatLngZoom(pos, 14));

        // --- LÓGICA DE ESTIMATIVA (BACKEND 2.2) ---
        if (t.isEstimado()) {
            txtAviso.setVisibility(View.VISIBLE);
        } else {
            txtAviso.setVisibility(View.GONE);
        }

        // --- CÁLCULO DE ETA (BACKEND 2.1) ---
        // O Backend já manda calculado!
        String textoEta = String.format("Chegada em: %d min %d seg", t.getEtaMinutos(), t.getEtaSegundos());
        txtEta.setText(textoEta);
    }
}