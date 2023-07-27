from typing import List, AnyStr

from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from .models import Device, Run, Report, Reservoir
import pandas as pd
from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np
import os
import glob
import math
from itertools import product
import ziontools
from sklearn import linear_model
import cv2 as cv
from scipy import ndimage
import roifile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
from django.utils import timezone
import dateutil.parser
from .tasks import run_pixel_extraction, mul
from .forms import RoiSetForm

from django.http import HttpResponseRedirect
from django.urls import reverse

# TODO, move to configuration
data_root_path = "/static_root/InstrumentData"
sequencing_file = '/tmp/Sequencing.csv'
reservoir_file = '/tmp/Reservoir_Inventory.csv'

def extract_datetime(subfolderstr: str) -> str | None:
    """TODO timestamp needs to be provided in iso format instead of being extracted from folder name"""
    match = re.search(r'^(20\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})', subfolderstr)

    if not match:
        print(f'unable to extract datetime from {subfolderstr}')
        return None

    # print(type(match))
    # print(match.groups())
    # print(match.group(0))

    datetime_str = match.group(0)  # 20230612_1145
    datetime_object = datetime.strptime(datetime_str, '%Y%m%d_%H%M')
    # print(datetime_object) # 2023-06-12 11:45:00

    aware_start_time = timezone.make_aware(datetime_object)  # aware

    return aware_start_time


def update_run_start_date():

    runs = Run.objects.all()
    for run in runs:
        if not run.started_at:
            d = extract_datetime(run.name)
            if d:
                run.started_at = d
                run.save()


def import_signup_sheet():

    if not os.path.exists(sequencing_file):
        print(f"{sequencing_file} not found")
        return

    df = pd.read_csv(sequencing_file)
    # drop rows that are fully empty
    df.dropna(how="all", inplace=True)

    for index, row in df.iterrows():
        print(row)

        # search runs for Seq ID
        # filter out QC
        # check if transformer matches, if not print error
        # update reservoir if not set, print error if different
        # add to notes: Scientist	Seq Ctrl Program	Description	Deviations

        runs = Run.objects.filter(name__contains=row['Seq ID'])

        # update() only works on querysets,
        if runs:
            print("type of run", type(runs))
            runs.update(notes=row['Description'])
            runs.update(operator=row['Scientist'])
            runs.update(sequencing_protocol=row['Seq Ctrl Program'])
            match = re.search("^R(\d{1,4})$", str(row['Reservoir']))
            if match:
                reservoir_sn = int(match.group(1))
            else:
                print(f"skip {row['Reservoir']}")
                continue
            reservoir = Reservoir.objects.filter(serial_number=reservoir_sn)
            print('received object', reservoir)
            if reservoir:
                print('run update')
                runs.update(reservoir=reservoir[0])  # TODO, make sure it is unique


def import_reservoirs_from_csv():

    if not os.path.exists(reservoir_file):
        print(f"{reservoir_file} not found")
        return

    df = pd.read_csv(reservoir_file)

    # show bad entries
    #df.dropna(subset=['Unnamed: 16'])

    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # drop rows that are fully empty
    df.dropna(how="all", inplace=True)

    '''
    created_at = models.DateTimeField(auto_now_add=True)
    serial_number = models.IntegerField(null=False, unique=True)
    vendor = models.CharField(max_length=200)
    substrate = models.CharField(max_length=200)
    cleaning = models.CharField(max_length=200)
    coating = models.CharField(max_length=200)
    functionalization = models.CharField(max_length=200)
    library = models.CharField(max_length=200)
    blocking = models.CharField(max_length=200)
    '''

    for index, row in df.iterrows():
        print(row)

        reservoir_sn = int(row['ID'])

        assembly_date = None
#        print(f"date: {row['Assembly Date']} type: {type(row['Assembly Date'])}")
        if isinstance(row['Assembly Date'], float):
            print(f"skip reservoir {reservoir_sn}, assembly_date is float: {row['Assembly Date']}")
        else:
            assembly_date = dateutil.parser.parse(row['Assembly Date'])

        reservoirs = Reservoir.objects.filter(serial_number=reservoir_sn)
        if reservoirs.exists():
#            reservoirs.update(assembly_date=assembly_date)
            continue

        r = Reservoir(
            assembly_date=assembly_date,
            serial_number=reservoir_sn,
            vendor=row['Slide Vendor'],
            substrate=row['Substrate'],
            cleaning=row['Cleaning'],
            coating=row['Coating'],
            functionalization=row['Functionalization'],
            library=row['SpotLayout#']
#            blocking=models.CharField(max_length=200)
        )
        r.save()


def scan():

    if not os.path.isdir(data_root_path):
        print(f"error with: {data_root_path}")
        return

    devices = Device.objects.all()

    for dev in devices:
        print(dev.name)

        raw_device_path = os.path.join(data_root_path, dev.name)
        if os.path.isdir(raw_device_path):

            subfolders: list[AnyStr] = [f.path for f in os.scandir(raw_device_path) if f.is_dir()]

            for subfolder in subfolders:
                run_directory = os.path.basename(subfolder)
                print(run_directory)

                # guess number of cycles:
                prot_file_names = sorted(glob.glob(subfolder + "/*Protocol*", recursive=False))
                if prot_file_names:
                    print(prot_file_names[-1])

                if Run.objects.filter(name=run_directory).exists():
                    # run exists already
                    continue

                # extract date from subfolder, TODO, should be in a json file, in ISO format
                started_at = extract_datetime(run_directory)

                r = Run(
                    name=run_directory,
                    device=Device.objects.get(name=dev.name),
                    path=subfolder.lstrip('/static_root'),
                    started_at=started_at,
#                    notes="test",
#                    reservoir=Reservoir.objects.get(serial_number=366),
                )
                print(f"Create new run {r}")
                r.save()

    '''

    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=200)
    device = models.ForeignKey(Device, null=True, blank=True, on_delete=models.SET_NULL)
    path = models.CharField(max_length=200, null=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    notes = models.CharField(max_length=400, null=True)
    reservoir = models.ForeignKey(Reservoir, null=True, blank=True, on_delete=models.SET_NULL)

    '''


def home(request):
    devices = Device.objects.all()

    scan()
#    update_run_start_date()
#    import_reservoirs_from_csv()
    import_signup_sheet()

    run_histogram = create_run_histogram()
    return render(request, 'home.html', {'devices': devices, 'run_histogram': run_histogram})

def test(request, run_id):

    r = Run.objects.get(id=run_id)
    print(f"Run: {r}")

    run_pixel_extraction.delay("some run info id ...")

    return render(request, 'test.html', {'req': request, 'post': request.POST})

def test2(request, run_id):

    r = Run.objects.get(id=run_id)
    print(f"Run: {r}")

    run_pixel_extraction.delay("some run info id ...")

    return render(request, 'test.html', {'req': request, 'post': request.POST})

'''
from django import forms
class FormsBasics(View):
    def get(selfself, request):
        form = RoiSetForm()
        context = {
            'form': form
        }
        return render(request, 'test.html', context)
    def post(selfself, request):
        form = RoiSetForm(request.POST)
        context = {
            'form': form
        }
        return render(request, 'test.html', context)
'''
class DeviceListView(generic.ListView):
    model = Device

class RunListView(generic.ListView):
    model = Run
    ordering = ['-name']

class ReportListView(generic.ListView):
    model = Report
    ordering = ['-created_at']

class ReservoirListView(generic.ListView):
    model = Reservoir
    ordering = ['-serial_number']


class DeviceDetailView(generic.DetailView):
    model = Device

class RunDetailView(generic.DetailView):
    model = Run

    # override context data
    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)
        context["form"] = RoiSetForm
        return context

    def post(self, request, *args, **kwargs):
        form = RoiSetForm(request.POST, request.FILES)
        print("received post -------------------------")
        print(kwargs)
        print(request.POST)
        print(request.FILES)

        run = Run.objects.get(pk=kwargs['pk'])
        print(f"post received from {run.id}")

        if form.is_valid():
            print("is valid  -------------------------")
            self.object = self.get_object()

            '''
                created_at = models.DateTimeField(auto_now_add=True)
                name = models.CharField(max_length=200)
                run = models.ForeignKey(Run, null=True, blank=True, on_delete=models.SET_NULL)
                path = models.CharField(max_length=200, null=True)
            
            '''

            report = Report(
                name=request.POST['title'],
                run=Run.objects.get(pk=kwargs['pk'])
            )
            report.save()

            print(f"report {report.pk} created =====================================")

            # create report directory
            try:
                mode = 0o777
                report_rel_path = "reports/" + str(report.pk)
                report_full_path = os.path.join("/static_root", report_rel_path)
                os.mkdir(report_full_path)
                os.chmod(report_full_path, mode)
            except OSError as error:
                print(error)

            report.path = report_rel_path
            report.save()

            print(request.FILES["file"])
            file1 = request.FILES['file']
            contentOfFile = file1.read()
#            print(contentOfFile)
            # permission issues? writes as www-data
            roisetzipfile = os.path.join(report_full_path, "roiset.zip")
            with open(roisetzipfile, "wb") as text_file:
                text_file.write(contentOfFile)

            # start celery task
            print(f"Run: {run.name} {run.id} {run.path}")
            run_full_path = os.path.join("/static_root", run.path, "raws")

            #            a = run_pixel_extraction.delay("/home/domibel/454_Bio/runs/20230527_1548_S0140_0001_OnePot10Cycles/raws", report_full_path, "roiset.zip")
            a = run_pixel_extraction.delay(run_full_path, report_full_path, "roiset.zip")

#            a.get()

            # context = super().get_context_data(**kwargs)
            # context['form'] = RoiSetForm
            return HttpResponseRedirect(reverse('dashboard:report-detail', args=[report.pk]))

        else:
            print("is not valid -------------------------")
            self.object = self.get_object()
            context = super().get_context_data(**kwargs)
            context['form'] = form
            return self.render_to_response(context=context)

    '''

        run = self.get_object()

        print(f"data_root_path: {data_root_path}")

        run_dir = os.path.join(data_root_path, run.device.name, run.name)
        print(f"run_path_tif: {run_dir}")
        tif_file_names = sorted(glob.glob(run_dir + "/raws/*.tif", recursive=False))
        print(tif_file_names)
        # create list of tuples (filename, full path)
        #        tif_file_names = [(os.path.basename(f), f.replace('/home/domibel/454_Bio/bio/dashboard/static/','')) for f in tif_file_names]
        tif_file_names = [(os.path.basename(f), f.replace('/static_root/', '')) for f in tif_file_names]
        print(tif_file_names)



        print("run.path: " + run.path, run_dir)
        file_names = sorted(glob.glob(run_dir + "/raws/*.jpg", recursive=False))
        file_names += sorted(glob.glob(run_dir + "/*.jpg", recursive=False))
        #file_names = sorted(glob.glob("/home/domibel/454_Bio/bio/dashboard/static/runs/" + run.path + "/raws/*.jpg", recursive=False))
        print("jpg paths:", file_names)
        # create list of tuples (filename, full path)
        #file_names = [(os.path.basename(f), f.replace('/home/domibel/454_Bio/bio/dashboard/static/','')) for f in file_names]
        file_names = [(os.path.basename(f), f.replace('/static_root/', '')) for f in file_names]
        print("jpg", file_names)

        # generate plotly graph
        # df = pd.read_csv('/home/domibel/454_Bio/results/79/rgbs.csv')
        # simple_plot = create_simple_plot_go(df)
        # context["simple_plot"] = simple_plot


        # add extra field
        context["raw_thumbnails"] = file_names


#        context["raw_tif"] = tif_file_names
        return context
    '''

class ReportDetailView(generic.DetailView):
    model = Report

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)

        report = self.get_object()
        report_full_path = os.path.join("/static_root", report.path)

        analysis_filenames = sorted(glob.glob(report_full_path + "/*.png", recursive=False))
        analysis_filenames = [(os.path.basename(f), f.replace('/static_root/', '')) for f in analysis_filenames]
        print("analysis_filenames: ", analysis_filenames)

        color_transformed_spots_csv = os.path.join(report_full_path, 'color_transformed_spots.csv')
        if os.path.exists(color_transformed_spots_csv):
            df = pd.read_csv(color_transformed_spots_csv)
            fig = ziontools.plot_bars(df, '') # Title TODO
            bar_plot = plot(fig, output_type='div')
            context["bar_plot"] = bar_plot

        basecalls_csv = os.path.join(report_full_path, 'basecalls.csv')
        if os.path.exists(basecalls_csv):
            try:
                df = pd.read_csv(basecalls_csv)
                print(df)
                df = df.dropna()
                context["basecalls"] = df.to_html(classes='table table-stripped')
                context["read_length_histogram"] = create_read_length_histogram(df)
                context["quality_score_graph"] = create_quality_score_graph(df)
            except pd.errors.EmptyDataError:
                print(f"ERROR parsing {basecalls_csv}")

        pixel_data_csv = os.path.join(report_full_path, 'spot_pixel_data.csv')
        if os.path.exists(pixel_data_csv):
            try:
                triangle_plot = create_spot_triangle_plot(pixel_data_csv)
                context["triangle_plot"] = triangle_plot
            except pd.errors.EmptyDataError:
                print(f"ERROR parsing {pixel_data_csv}")

        metrics_data_csv = os.path.join(report_full_path, 'metrics.csv')
        if os.path.exists(metrics_data_csv):
            fig = ziontools.plot_spot_trajectories(
                metrics_data_csv, ['R365', 'G365', 'B365', 'R445'], ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
            )
            spot_trajectories_plot = plot(fig, output_type='div')
            context["spot_trajectories_plot"] = spot_trajectories_plot

            df = pd.read_csv(metrics_data_csv)
            #context["df_metrics"] = df.to_html()

            # for each spot in list produce a plot
            if 'spot_index' in df:
                spot_indizes = df.spot_index.unique()
                context["spot_intensities_vs_time"] = [plot_spot_intensities(df, spot_index) for spot_index in spot_indizes]

        context["analysis_filenames"] = analysis_filenames


        return context

def create_spot_triangle_plot(pixel_data_csv: str):

    fig = ziontools.plot_triangle(
        pixel_data_csv, ['R365', 'G365', 'B365', 'R445'], None, None
    )

    plot1 = plot(fig, output_type='div')
    return plot1

def create_simple_plot_go(df_: pd.DataFrame):

    print(df_)
    fig = go.Figure()

    for wavelength in [0, 645, 590, 525, 445]:
        df = df_.loc[(df_['WL'] == wavelength) & (df_['spot'] == 1)]
        df = df.sort_values(by=['TS'])

        fig.add_trace(go.Scatter(x=df['cycle'], y=df['Gavg'],    mode='lines', name='G_'+str(wavelength),  opacity=0.7, marker_color='red'))
        fig.add_trace(go.Scatter(x=df['cycle'], y=df['Ravg'],    mode='lines', name='R_'+str(wavelength),  opacity=0.7, marker_color='green'))
        fig.add_trace(go.Scatter(x=df['cycle'], y=df['Bavg'],    mode='lines', name='B_'+str(wavelength),  opacity=0.7, marker_color='blue'))

    fig.update_layout(
        title_text="Spot 0",
        title_font_size=30,
        xaxis={'title': 'Cycle #'},
        yaxis={'title': '16 bit intensity'},
        yaxis2={'title': 'Temp C and Voltage V', 'overlaying': 'y', 'side': 'right'},
        height=800
    )

    plot1 = plot(fig, output_type='div')
    return plot1


def create_run_histogram():

    df = pd.DataFrame(list(Run.objects.exclude(name__contains="QC|qc").values()))
    df = df[~df.name.str.contains("QC|qc")]
#    print(df["name"])

    fig = go.Figure(
        go.Histogram(
            x=df['started_at'],
            y=df['id'],
            histnorm='',
            histfunc='count',
            autobinx=False,
            xbins=dict(
                start=datetime.now().date()-timedelta(days=100),
                end=datetime.now().date(),
                size='D'
            )
        )
    )
    fig.update_layout(bargap=0.1)
    fig.show()

    plot1 = plot(fig, output_type='div')
    return plot1

def create_read_length_histogram(df_: pd.DataFrame):

    if not {'length_perfect_max_intensity', 'length_perfect_phase_corrected'}.issubset(df_.columns):
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_['length_perfect_max_intensity'], name='max_intensity', xbins_size=1))
    fig.add_trace(go.Histogram(x=df_['length_perfect_phase_corrected'], name='phase_corrected', xbins_size=1))
    fig.add_vline(
        x=np.mean(df_['length_perfect_max_intensity']), line_dash='dash', line_color='blue',
        annotation_text="mean", annotation=dict(font_size=23, font_family="Times New Roman")
    )
    fig.add_vline(
        x=np.mean(df_['length_perfect_phase_corrected']), line_dash='dash', line_color='red',
        annotation_text="mean", annotation=dict(font_size=23, font_family="Times New Roman")
    )

    # Overlay both histograms
    fig.update_layout(
        barmode='overlay',
        title_text='Read-length histogram of perfect reads',  # title of plot
        xaxis_title_text='Read length',  # xaxis label
        yaxis_title_text='Count',  # yaxis label
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
    )

    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.40)
    fig.show()
    plot1 = plot(fig, output_type='div')
    return plot1


def create_quality_score_graph(df: pd.DataFrame) -> go.Figure | None:

    if 'expected_read' not in df:
        return None

    result_df = df[df['expected_read'].notna()]
    print(result_df)
    numSpots = len(result_df)

    if numSpots == 0:
        print(f"Error: empty dataframe, len(df)={len(df)}, len(result_df)={len(result_df)}")
        return None

    numCycles = result_df.iloc[0]['cycles']
    quality_df = pd.DataFrame(columns=['cycle', 'Q_color_transform', 'Q_dephased'])
    for cycle in range(numCycles):
        p_color_transform = 0
        p_dephased = 0
        for index, spot_row in result_df.iterrows():
            curr_color_transform_basecall = spot_row['max_intensity_read'][:(cycle + 1)]
            curr_dephased_basecall = spot_row['phase_corrected_read'][:(cycle + 1)]
            curr_truth_basecall = spot_row['expected_read'][:(cycle + 1)]

            print("index:", index, curr_truth_basecall, curr_color_transform_basecall, curr_dephased_basecall)

            p_color_transform += sum(char1 != char2 for char1, char2 in zip(curr_color_transform_basecall, curr_truth_basecall))
            p_dephased += sum(char1 != char2 for char1, char2 in zip(curr_dephased_basecall, curr_truth_basecall))

        p_color_transform = p_color_transform / ((cycle + 1) * numSpots)
        p_dephased = p_dephased / ((cycle + 1) * numSpots)

        if p_color_transform == 0:
            p_color_transform = 0.01
        if p_dephased == 0:
            p_dephased = 0.01
        print(p_color_transform, p_dephased)

        Q_color_transform = -10 * math.log10(p_color_transform)
        Q_dephased = -10 * math.log10(p_dephased)
        quality_df.at[cycle, 'p_dephased'] = p_dephased
        quality_df.at[cycle, 'p_color_transform'] = p_color_transform
        quality_df.at[cycle, 'cycle'] = cycle + 1
        quality_df.at[cycle, 'Q_color_transform'] = Q_color_transform
        quality_df.at[cycle, 'Q_dephased'] = Q_dephased

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=quality_df['cycle'], y=quality_df['Q_color_transform'], name="Q max intensity"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=quality_df['cycle'], y=quality_df['Q_dephased'], name="Q phase corrected"),
        secondary_y=False,
    )
    '''
    fig.add_trace(
        go.Scatter(x=quality_df['cycle'], y=quality_df['p_color_transform'], name="P max intensity"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=quality_df['cycle'], y=quality_df['p_dephased'], name="P phase corrected"),
        secondary_y=True,
    )
    '''

    fig.update_traces(textposition="top center")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Q score<b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Error probability</b>", secondary_y=True)

    fig.update_xaxes(title_text='Cycles')


    fig.show()
    plot1 = plot(fig, output_type='div')
    return plot1


def plot_spot_intensities(df: pd.DataFrame, spot_index) -> go.Figure | None:

    # df['Ravg'] -= 4096
    # df['Gavg'] -= 4096
    # df['Bavg'] -= 4096

    excitations = [445, 525, 590, 645, 365]
    image_channels = ['R', 'G', 'B']

    channels = list(product(image_channels, excitations))
    print(len(channels), "channels: ", channels)

    df_spot = df.loc[(df['spot_index'] == spot_index)]
    spot_name = df_spot.spot_name.unique()[0]

    fig = go.Figure()

    for c, channel in enumerate(channels):
        dft = df.loc[(df['spot_index'] == spot_index) & (df['WL'] == channel[1])]
        X = dft['TS']
        channelname = channel[0] + 'avg'
        Y = dft[channelname]

        fig.add_trace(
            go.Scatter(
                x=X,
                y=Y,
                mode="markers+lines",
                name=channel[0] + str(channel[1]),
            )
        )

    fig.update_layout(
        title={
            'text': str(spot_index) + ' ' + spot_name + "  (" + ziontools.common.oligo_sequences.get(spot_name, "")[:16] + ")",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        autosize=False,
        width=2000,
        height=1000,
        xaxis_title="timestamp ms",
        yaxis_title="Intensity 16bit uncorrected",
        legend_title="15 channels",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    fig.show()
    plot1 = plot(fig, output_type='div')
    return plot1
